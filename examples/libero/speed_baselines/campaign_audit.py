"""Terminal audit of complete banks and the actual applied action groups."""

from __future__ import annotations

import json
import math

import numpy as np

from .campaign_policy import array_hash
from .campaign_routes import SUITE_HORIZONS


def audit_episode(rows, result, *, route, composer, expected_reference=None):
    def require(value, message):
        if not value:
            raise ValueError(message)

    require(bool(rows), "missing episode evidence")
    episode = result["episode_index"]
    require(all(r["episode"] == episode for r in rows), "mixed episode evidence")
    initial = [r for r in rows if r["kind"] == "initial"]
    terminal = [r for r in rows if r["kind"] == "terminal"]
    scenes = [r for r in rows if r["kind"] == "scene"]
    require(len(initial) == len(terminal) == len(scenes) == 1, "missing or duplicate boundary evidence")
    steps = [r for r in rows if r["kind"] == "step"]
    plans = [r for r in rows if r["kind"] == "plan"]
    require([s["env_step"] for s in steps] == list(range(result["env_steps"])), "control tick trace incomplete")
    require(terminal[0]["env_steps"] == result["env_steps"], "terminal tick mismatch")
    require(type(result["success"]) is bool and terminal[0]["success"] == result["success"], "success mismatch")
    require(terminal[0]["state"]["goal_satisfied"] == result["success"], "official goal mismatch")
    require(all(terminal[0]["state"]["goal_predicates"].values()) == result["success"], "goal conjunction mismatch")
    require(result["success"] or result["env_steps"] == SUITE_HORIZONS[route.identity.suite], "failure stopped early")
    wall = result["loop_wall_seconds"]
    require(math.isfinite(wall) and wall >= 0 and wall == terminal[0]["loop_wall_seconds"], "invalid loop walltime")
    require(len(plans) == result["decision_count"], "policy call count mismatch")
    require([p["decision"]["decision_index"] for p in plans] == list(range(len(plans))), "plan order mismatch")
    by_plan = {}
    for step in steps:
        by_plan.setdefault(step["decision_index"], []).append(step)
    total_source = 0
    for index, plan in enumerate(plans):
        receipt = plan["receipt"]
        require(receipt["episode"] == episode and receipt["plan"] == index, "RNG receipt episode/plan mismatch")
        require(receipt["seed"] == 0, "policy RNG seed differs")
        require(
            receipt["rng_before"] == ([0, 0] if index == 0 else plans[index - 1]["receipt"]["rng_after"]),
            "policy RNG reset/continuation mismatch",
        )
        require(
            receipt["suite"] == route.identity.suite
            and receipt["task"] == route.identity.task_id
            and receipt["bddl_sha256"] == route.identity.bddl_sha256,
            "RNG receipt task mismatch",
        )
        raw = np.asarray(receipt["raw_actions"], dtype=receipt["actions_dtype"])
        require(raw.shape == (10, 7) and np.isfinite(raw).all(), "invalid raw policy actions")
        require(array_hash(raw) == receipt["actions_sha256"], "raw action receipt corruption")
        cursor = 0
        require(index in by_plan, "unused recorded policy plan")
        for step in by_plan[index]:
            indices = step["source_indices"]
            require(
                indices == list(range(cursor, cursor + len(indices))) and bool(indices), "source action skip or reuse"
            )
            require(indices[-1] < len(raw) and len(indices) == step["source_stride"], "source group bounds mismatch")
            if route.method == "uniform":
                require(
                    indices == list(range(cursor, min(cursor + route.uniform_stride, len(raw)))),
                    "incorrect uniform stride",
                )
            else:
                speeds = plan["decision"]["speed_factors"]
                require(len(speeds) == 10 and all(v in (1, 2, 3) for v in speeds), "unregistered STRIDER speeds")
                stop = min(cursor + int(speeds[cursor]), len(raw))
                if not all(v == speeds[cursor] for v in speeds[cursor:stop]):
                    stop = cursor + 1
                require(indices == list(range(cursor, stop)), "STRIDER group differs from phase schedule")
            group = raw[indices].astype(np.float64)
            action = group[0] if len(group) == 1 else composer.merge(group)
            require(np.allclose(action, step["action"], rtol=0, atol=1e-12), "applied action composition mismatch")
            cursor += len(indices)
            total_source += len(indices)
        require(index == len(plans) - 1 or cursor == 10, "new inference discarded a raw suffix")
    require(total_source == result["source_actions_consumed"], "source action total mismatch")
    if route.method == "uniform":
        require(not result.get("phase_decisions"), "uniform run executed a phase selector")
    else:
        require(len(result["phase_decisions"]) == len(steps), "STRIDER phase trace incomplete")
    if expected_reference is not None:
        require(
            initial[0]["observations"] == expected_reference["observations"], "restored initial observations differ"
        )
        require(bool(plans), "no initial policy receipt for matched comparison")
        for key in ("request_sha256", "actions_sha256", "rng_before", "rng_after"):
            require(
                plans[0]["receipt"][key] == expected_reference["first_receipt"][key], f"restored first {key} differs"
            )
    return {
        "episode": episode,
        "passed": True,
        "success": result["success"],
        "env_steps": len(steps),
        "source_actions": total_source,
        "policy_calls": len(plans),
        "loop_wall_seconds": wall,
    }


def audit_run(session, *, composer):
    payload = json.loads((session.run_dir / "results.json").read_text())
    if payload["status"] != "complete" or [e["episode_index"] for e in payload["episodes"]] != list(session.bank):
        raise ValueError("campaign bank incomplete or reordered")
    references = session.spec.get("reference_initials", {})
    if session.scene_mode == "restore" and set(references) != {str(ep) for ep in session.bank}:
        raise ValueError("restore requires pinned initial observation/inference references")
    checked = []
    for result in payload["episodes"]:
        ep = result["episode_index"]
        rows = [json.loads(line) for line in (session.output / f"episode-{ep}.jsonl").read_text().splitlines()]
        checked.append(
            audit_episode(
                rows,
                result,
                route=session.policy_spec.route,
                composer=composer,
                expected_reference=references.get(str(ep)),
            )
        )
    return {
        "passed": True,
        "session_sha256": session.sha256,
        "episodes": checked,
        "scope": "Complete registered bank, exact applied action composition, official goal traces and matched initial inputs",
    }
