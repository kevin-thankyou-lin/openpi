"""Optional, fail-closed campaign lifecycle around the unchanged speed controller.

One process owns one complete ordered task/arm bank. A consumed start is never
automatically retried. The dispatcher owns GPU exclusion, server lifetime and
artifact mirroring; this module owns episode accounting and physical evidence.
"""

# LIBERO exposes its authoritative goal and grasp checks through private methods.
# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .campaign_budget import BudgetError
from .campaign_budget import CampaignBudget
from .campaign_budget import Cap
from .campaign_policy import EpisodePolicy
from .campaign_policy import array_hash
from .campaign_policy import load_policy_spec
from .campaign_policy import preflight
from .campaign_policy import validate_metadata
from .campaign_provenance import verify_controller_identity
from .campaign_provenance import verify_phase_runtime
from .campaign_routes import FileAsset
from .scene_replay import SceneIdentity
from .scene_replay import capture_scene
from .scene_replay import load_scene
from .scene_replay import observation_fingerprints
from .scene_replay import restore_scene
from .scene_replay import write_scene

SCHEMA = "strider-libero-session-v1"
CHALLENGER_PURPOSES = ("search_candidate", "causal_replay", "candidate_validation")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def physical_state(env, obs):
    base = env
    while not hasattr(base, "parsed_problem"):
        base = base.env
    goals = base.parsed_problem["goal_state"]
    predicates = {" ".join(goal): bool(base._eval_predicate(goal)) for goal in goals}
    success = bool(base._check_success())
    if success != all(predicates.values()) or not predicates:
        raise ValueError("BDDL predicates disagree with official task success")
    contacts = [
        {
            "geom1": base.sim.model.geom_id2name(contact.geom1),
            "geom2": base.sim.model.geom_id2name(contact.geom2),
            "distance": float(contact.dist),
        }
        for contact in base.sim.data.contact[: base.sim.data.ncon]
    ]
    objects = {}
    for names in base.parsed_problem["objects"].values():
        for name in names:
            obj = base.get_object(name)
            body = base.sim.model.body_name2id(obj.root_body)
            objects[name] = {
                "position": np.asarray(base.sim.data.body_xpos[body]).tolist(),
                "quaternion_wxyz": np.asarray(base.sim.data.body_xquat[body]).tolist(),
                "grasped": bool(base._check_grasp(gripper=base.robots[0].gripper, object_geoms=obj.contact_geoms)),
            }
    return {
        "goal_predicates": predicates,
        "goal_satisfied": success,
        "objects": objects,
        "contacts": contacts,
        "eef_position": np.asarray(obs["robot0_eef_pos"]).tolist(),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
        "qpos": np.asarray(base.sim.data.qpos).tolist(),
        "qvel": np.asarray(base.sim.data.qvel).tolist(),
    }


class CampaignSession:
    def __init__(self, spec_path, *, repo):
        self.path = Path(spec_path).resolve()
        self.spec = json.loads(self.path.read_text())
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.spec.get("schema") != SCHEMA:
            raise ValueError("unsupported campaign session schema")
        self.repo = Path(repo).resolve()

        def asset(value):
            item = FileAsset((self.path.parent / value["path"]).resolve(), value["sha256"])
            item.verify()
            return item

        policy_asset = asset(self.spec["policy_spec"])
        self.policy_spec = load_policy_spec(policy_asset.path)
        preflight(self.policy_spec, repo=self.repo)
        self.bank = self.policy_spec.ordered_episode_indices
        self.identity = self.policy_spec.route.identity
        if (
            self.spec["source_pins"]["policy_sha256"] != self.policy_spec.policy_sha256
            or self.spec["source_pins"]["controller_sha256"] != self.policy_spec.controller_sha256
        ):
            raise ValueError("ledger/scene policy or controller identity differs from verified policy spec")
        if self.spec["purpose"] not in (*CHALLENGER_PURPOSES, "baseline"):
            raise ValueError("explicit rollout purpose required")
        if self.spec["purpose"] == "baseline" and self.policy_spec.route.method != "uniform":
            raise ValueError("a learned/gated challenger cannot be exempted as a baseline")
        if type(self.spec["baseline_cap"]) is not int or self.spec["baseline_cap"] < 0:
            raise ValueError("explicit baseline allocation required")
        if self.spec["challenger_cap"] != 20 or type(self.spec["challenger_cap"]) is not int:
            raise ValueError("this campaign has a hard 20-episode challenger cap")
        self.sources = {key: asset(value) for key, value in self.spec["implementation_sources"].items()}
        if (
            not {"session", "evaluator", "actions", "controller", "policy_adapter", "scene_replay"}
            <= self.sources.keys()
        ):
            raise ValueError("missing executable source pins")
        expected_files = {
            "session": Path(__file__),
            "evaluator": Path(__file__).with_name("main.py"),
            "actions": Path(__file__).with_name("actions.py"),
            "controller": Path(__file__).with_name("controller.py"),
            "policy_adapter": Path(__file__).with_name("campaign_policy.py"),
            "scene_replay": Path(__file__).with_name("scene_replay.py"),
            "budget": Path(__file__).with_name("campaign_budget.py"),
            "routes": Path(__file__).with_name("campaign_routes.py"),
            "provenance": Path(__file__).with_name("campaign_provenance.py"),
            "audit": Path(__file__).with_name("campaign_audit.py"),
            "strider_client": Path(__file__).with_name("strider_client.py"),
            "acquisition_window": Path(__file__).with_name("acquisition_window.py"),
            "telemetry": Path(__file__).with_name("telemetry.py"),
            "eval_entrypoint": self.repo / "scripts/eval_campaign_libero.py",
            "server_entrypoint": self.repo / "scripts/serve_campaign_libero.py",
        }
        if self.policy_spec.route.method == "strider":
            expected_files["phase_preprocessing"] = Path(__file__).with_name("phase_preprocessing.py")
        if not expected_files.keys() <= self.sources.keys():
            raise ValueError("missing executable source pins")
        if any(self.sources[key].path != path.resolve() for key, path in expected_files.items()):
            raise ValueError("source pins must name the actual running implementation")
        self.states = self.spec["initial_states"]
        if set(self.states) != {str(ep) for ep in self.bank}:
            raise ValueError("initial state bank differs from policy bank")
        self.scene_mode = self.spec["scene_mode"]
        if self.scene_mode not in ("capture", "restore"):
            raise ValueError("explicit capture or restore mode required")
        if self.scene_mode == "capture" and self.spec["purpose"] != "baseline":
            raise ValueError("challengers must restore matched baseline scenes")
        self.scenes = {}
        for ep in self.bank:
            self.scene_identity(ep)
        if self.scene_mode == "restore":
            if set(self.spec.get("reference_initials", {})) != set(self.states):
                raise ValueError("restore requires pinned initial observation/inference references")
            if set(self.spec["scenes"]) != set(self.states):
                raise ValueError("snapshot bank differs from initial state bank")
            for ep in self.bank:
                item = asset(self.spec["scenes"][str(ep)])
                self.scenes[ep] = load_scene(
                    item.path, expected_sha256=item.sha256, expected_identity=self.scene_identity(ep)
                )
        self.run_dir = self.policy_spec.run_dir
        self.reservation = self.spec["unit_id"]
        self.budget = None
        self.reserved = False
        self.active = None
        self.seen = []
        self.stream = None

    def scene_identity(self, episode):
        pins = self.spec["source_pins"]
        return SceneIdentity(
            self.identity.suite,
            self.identity.task_id,
            episode,
            self.identity.bddl_sha256,
            self.states[str(episode)]["sha256"],
            pins["policy_sha256"],
            pins["controller_sha256"],
        )

    def validate_args(self, args):
        route = self.policy_spec.route
        if (
            args.task_suite_name != self.identity.suite
            or args.task_start != self.identity.task_id
            or args.task_count != 1
            or args.num_trials_per_task != len(self.bank)
            or args.method != route.method
            or args.seed != 7
            or args.chunk_size != 10
            or args.num_steps_wait != 10
            or args.resize_size != 224
            or args.resume
            or args.run_dir.resolve() != self.run_dir
        ):
            raise ValueError("evaluator arguments differ from frozen campaign contract")
        if route.method == "uniform" and args.uniform_stride != route.uniform_stride:
            raise ValueError("uniform stride differs from registered arm")
        if route.method == "strider" and (
            args.strider_checkpoint.resolve() != route.phase_checkpoint.path
            or args.strider_schedule.resolve() != self.policy_spec.rendered_schedule.path
            or args.fast_stride != 3
        ):
            raise ValueError("STRIDER assets differ from registered arm")
        if route.method == "strider":
            if args.strider_legacy_input_mode:
                raise ValueError("campaign forbids historical phase input mode")
            route.phase_input_binding()
            phase = self.spec["phase_runtime"]
            verify_phase_runtime(
                root=phase["root"],
                files=phase["files"],
                actual_root=args.strider_phase_repo.resolve() / "src",
                expected_sha256=phase["sha256"],
            )

    def validate_controller(self, composer):
        # Same OSC scaling as the frozen SuP path, checked against the actually
        # loaded robosuite configuration rather than trusting a source hash.
        expected = {
            "input_min": [-1.0] * 6,
            "input_max": [1.0] * 6,
            "output_min": [-0.05] * 3 + [-0.5] * 3,
            "output_max": [0.05] * 3 + [0.5] * 3,
        }
        for name, value in expected.items():
            if not np.array_equal(getattr(composer, name), value):
                raise ValueError(f"actual OSC action scale differs: {name}")
        self.controller_receipt = verify_controller_identity(
            files={k: dataclasses.asdict(self.sources[k]) for k in ("actions", "controller")},
            actual_files={k: Path(__file__).with_name(k + ".py") for k in ("actions", "controller")},
            control_hz=20,
            chunk_size=10,
            expected_sha256=self.spec["source_pins"]["controller_sha256"],
        )

    def write_phase_input_proof(self, proof):
        if self.policy_spec.route.method != "strider" or proof is None:
            raise ValueError("checked phase proof required")
        self.policy_spec.route.verify_phase_input_proof(proof)
        write_new(self.output / "phase-input-proof.json", proof)

    def connect(self, metadata):
        validate_metadata(metadata, self.policy_spec)
        # Reject an existing unit before reserving; no implicit resume or overwrite.
        self.output = self.run_dir / "campaign"
        self.output.mkdir()
        (self.output / "scenes").mkdir()
        write_new(self.output / "session.json", {"spec": self.spec, "sha256": self.sha256})
        self.budget = CampaignBudget(
            self.path.parent / self.spec["ledger_path"],
            campaign_id=self.spec["campaign_id"],
            suite=self.identity.suite,
            task=self.identity.task_id,
            bddl_sha256=self.identity.bddl_sha256,
            source_pins=self.spec["source_pins"],
        )
        self.budget.configure(Cap("episodes", 20, CHALLENGER_PURPOSES), authority=self.spec["cap_authority"])
        self.budget.configure(
            Cap("episodes", self.spec["baseline_cap"], ("baseline",)),
            authority=self.spec["baseline_authority"],
            name="baselines",
        )
        self.budget.reserve(
            self.reservation,
            candidate_id=self.spec["candidate_id"],
            purpose=self.spec["purpose"],
            max_starts=len(self.bank),
        )
        self.reserved = True

    def validate_task(self, task, initial_states, bddl_path):
        if (
            task.language != self.policy_spec.prompt
            or hashlib.sha256(Path(bddl_path).read_bytes()).hexdigest() != self.identity.bddl_sha256
        ):
            raise ValueError("actual benchmark task differs from route")
        for ep in self.bank:
            a = np.asarray(initial_states[ep])
            expected = self.states[str(ep)]
            if expected != {"sha256": array_hash(a), "dtype": str(a.dtype), "shape": list(a.shape)}:
                raise ValueError(f"official initial state differs for episode {ep}")

    def before_episode(self, episode, policy):
        if self.budget is None or self.active is not None or self.bank[len(self.seen)] != episode:
            raise ValueError("episode lifecycle/order mismatch")
        start_id = f"{self.reservation}:episode:{episode}"
        if not self.budget.start(self.reservation, start_id=start_id, episode=episode):
            raise BudgetError("start already consumed; do not execute it again")
        self.active = start_id
        self.seen.append(episode)
        self.episode = episode
        self.stream = (self.output / f"episode-{episode}.jsonl").open("x", buffering=1)
        self.episode_policy = EpisodePolicy(policy, self.policy_spec, episode)
        return self.episode_policy

    def record(self, kind, **value):
        self.stream.write(json.dumps({"kind": kind, "episode": self.episode, **value}, allow_nan=False) + "\n")

    def after_set_init_state(self, env, obs, *, initial_state):
        base = env
        while not hasattr(base, "control_timestep"):
            base = base.env
        if base.control_timestep != 0.05:
            raise ValueError("actual environment is not running at 20 Hz")
        identity = self.scene_identity(self.episode)
        if self.scene_mode == "restore":
            scene = self.scenes[self.episode]
            obs = restore_scene(env, scene, expected_identity=identity, initial_state=initial_state)
        else:
            scene = capture_scene(env, obs, identity=identity, initial_state=initial_state)
        path = self.output / "scenes" / f"episode-{self.episode}.json"
        sha = write_scene(path, scene, expected_identity=identity)
        self.record("scene", file=str(path), sha256=sha, identity=dataclasses.asdict(identity), mode=self.scene_mode)
        return obs

    def after_warmup(self, env, obs):
        self.initial_observations = observation_fingerprints(obs)
        self.record("initial", observations=self.initial_observations, state=physical_state(env, obs))
        self.started = time.perf_counter()

    def before_action(self, env, obs, action_slice, env_step, decision):
        self.record(
            "step",
            env_step=env_step,
            state=physical_state(env, obs),
            action=np.asarray(action_slice.action).tolist(),
            source_indices=list(action_slice.source_indices),
            source_stride=action_slice.stride,
            metadata=action_slice.metadata,
            decision_index=decision["decision_index"],
        )
        if decision["env_step"] == env_step:
            self.record("plan", env_step=env_step, decision=decision, receipt=self.episode_policy.receipt)

    def after_loop(self, env, obs, success, env_steps):
        self.loop_wall_seconds = time.perf_counter() - self.started
        state = physical_state(env, obs)
        if state["goal_satisfied"] != bool(success):
            raise ValueError("terminal success disagrees with exact BDDL goals")
        self.record(
            "terminal",
            success=bool(success),
            env_steps=env_steps,
            state=state,
            loop_wall_seconds=self.loop_wall_seconds,
        )

    def finish_episode(self, result):
        result.update(
            suite=self.identity.suite,
            bddl_sha256=self.identity.bddl_sha256,
            initial_observations=self.initial_observations,
            loop_wall_seconds=self.loop_wall_seconds,
            campaign_session_sha256=self.sha256,
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.stream = None
        self.budget.outcome(self.active, "success" if result["success"] else "failure")
        self.active = None

    def close(self):
        """Call only after the evaluator's environment has closed in its finally block."""
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None
        if self.budget is not None and self.reserved:
            self.budget.reconcile(self.reservation, worker_stopped=True)
            write_new(self.output / "budget-final.json", self.budget.snapshot())
            self.reserved = False
