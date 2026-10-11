from __future__ import annotations

import ast
import copy
import dataclasses
import json
import pathlib

import pytest

from examples.libero.speed_baselines import campaign_routes as routes
from examples.libero.speed_baselines import strider_search_runner as runner


def _registry(tmp_path, *, suite="libero_spatial", method="strider", task_id=0):
    folder = tmp_path / suite
    folder.mkdir(exist_ok=True)
    bddl = folder / "task.bddl"
    bddl.write_text(f"task in {suite}")
    checkpoint = folder / "predictor.pt"
    checkpoint.write_bytes(b"opaque checkpoint; loader must not deserialize it")
    ontology = folder / "ontology.json"
    ontology.write_text(json.dumps({"task_id": task_id, "phases": ["approach", "grasp"]}))

    def asset(path):
        return {"path": str(path.relative_to(tmp_path)), "sha256": routes.file_sha256(path)}

    dataset = folder / "dataset.json"
    dataset.write_text(json.dumps({"schema":"strider-libero-prepared-phase-dataset-v1",
        "identity":{"suite":suite,"task_id":task_id,"bddl_sha256":routes.file_sha256(bddl)},
        "preprocessing":routes.PREPROCESSING,"classes":[{"name":"approach"},{"name":"grasp"}]}))
    entry = {"suite": suite, "task_id": task_id, "bddl": asset(bddl), "method": method}
    if method == "uniform":
        entry["uniform_stride"] = 2
    else:
        entry.update(
            phase_checkpoint=asset(checkpoint),
            ontology=asset(ontology),
            dataset_manifest=asset(dataset),
            phase_names=["approach", "grasp"],
            asset_binding={"suite": suite, "task_id": task_id, "bddl_sha256": entry["bddl"]["sha256"]},
        )
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"schema": routes.SCHEMA, "routes": [entry]}))
    return path, entry


def _save(path, entries):
    path.write_text(json.dumps({"schema": routes.SCHEMA, "routes": entries}))


def test_suite_horizons_match_existing_evaluator_without_importing_simulator():
    main = pathlib.Path(runner.__file__).with_name("main.py")
    node = next(
        n for n in ast.parse(main.read_text()).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAX_STEPS" for t in n.targets)
    )
    assert ast.literal_eval(node.value) == dict(routes.SUITE_HORIZONS)
    assert set(routes.SUITE_TASK_COUNTS) == set(routes.SUITE_HORIZONS)
    assert sum(routes.SUITE_TASK_COUNTS[s] for s in routes.DEFAULT_CAMPAIGN_SUITES) == 40


@pytest.mark.parametrize("suite, task", [("libero_spatial", 10), ("libero_90", 90), ("libero_10", -1),
                                         ("libero_object", True), ("libero_goal", 1.0), ("unknown", 0)])
def test_task_bounds_reject_invalid_identifiers(suite, task):
    with pytest.raises(ValueError):
        routes.TaskIdentity(suite, task, "a" * 64)


def test_uniform_routes_need_no_predictor_and_arms_are_distinct(tmp_path):
    path, entry = _registry(tmp_path, method="uniform")
    for name in ["predictor.pt", "ontology.json"]:
        (tmp_path / "libero_spatial" / name).unlink()
    entries = [{**entry, "uniform_stride": speed} for speed in (1, 2, 3)]
    _save(path, entries)
    loaded = routes.load_routes(path)
    assert len(loaded) == 3
    assert all(r.phase_checkpoint is None for r in loaded.values())


def test_same_task_index_in_different_suites_never_collides(tmp_path):
    path, spatial = _registry(tmp_path, method="uniform")
    _, goal = _registry(tmp_path, suite="libero_goal", method="uniform")
    _save(path, [spatial, goal])
    loaded = routes.load_routes(path)
    a = routes.select_route(loaded, suite="libero_spatial", task_id=0, arm="uniform2")
    b = routes.select_route(loaded, suite="libero_goal", task_id=0, arm="uniform2")
    assert a.identity.key != b.identity.key
    with pytest.raises(ValueError, match="missing route"):
        routes.select_route(loaded, suite="libero_object", task_id=0, arm="uniform2")


def test_support_for_90_is_not_campaign_authorization(tmp_path):
    path, _ = _registry(tmp_path, suite="libero_90", method="uniform", task_id=89)
    with pytest.raises(ValueError, match="not authorized"):
        routes.load_routes(path)
    assert len(routes.load_routes(path, allowed_suites=("libero_90",))) == 1


@pytest.mark.parametrize("field", ["phase_checkpoint", "ontology", "asset_binding", "dataset_manifest"])
def test_challenger_requires_explicit_asset_binding(tmp_path, field):
    path, entry = _registry(tmp_path)
    del entry[field]
    _save(path, [entry])
    with pytest.raises(ValueError):
        routes.load_routes(path)


@pytest.mark.parametrize("field,value", [("suite", "libero_goal"), ("task_id", 1), ("bddl_sha256", "0" * 64)])
def test_challenger_rejects_cross_task_binding(tmp_path, field, value):
    path, entry = _registry(tmp_path)
    entry["asset_binding"][field] = value
    _save(path, [entry])
    with pytest.raises(ValueError, match="binding"):
        routes.load_routes(path)


@pytest.mark.parametrize("field", ["bddl", "phase_checkpoint", "ontology", "dataset_manifest"])
def test_asset_mutation_rejected_at_load_and_reselection(tmp_path, field):
    path, entry = _registry(tmp_path)
    loaded = routes.load_routes(path)
    (tmp_path / entry[field]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        routes.load_routes(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        routes.select_route(loaded, suite="libero_spatial", task_id=0, arm="strider")


def test_duplicate_and_ontology_mismatch_rejected(tmp_path):
    path, entry = _registry(tmp_path)
    _save(path, [entry, copy.deepcopy(entry)])
    with pytest.raises(ValueError, match="duplicate"):
        routes.load_routes(path)
    entry["phase_names"] = ["wrong"]
    _save(path, [entry])
    with pytest.raises(ValueError, match="ontology phase"):
        routes.load_routes(path)


@pytest.mark.parametrize("speed", [True, 0, 4, 1.5])
def test_uniform_max_stride_is_three(tmp_path, speed):
    path, entry = _registry(tmp_path, method="uniform")
    entry["uniform_stride"] = speed
    _save(path, [entry])
    with pytest.raises(ValueError, match="stride"):
        routes.load_routes(path)


def _runner_args(tmp_path, path, entry):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps({"approach": 3, "grasp": 1}))
    return runner.Args(
        run_dir=tmp_path, schedule=schedule, phase_checkpoint=tmp_path / entry["phase_checkpoint"]["path"],
        phase_repo=tmp_path, server_python=pathlib.Path(__file__), task_id=0,
        phase_names="approach,grasp", task_suite_name="libero_spatial", route_registry=path,
    )


def test_runner_legacy_default_and_nonlegacy_requires_route(tmp_path):
    path, entry = _registry(tmp_path)
    args = _runner_args(tmp_path, path, entry)
    assert runner.Args.__dataclass_fields__["task_suite_name"].default == "libero_10"
    with pytest.raises(ValueError, match="require.*route_registry"):
        runner._validate_args(dataclasses.replace(args, route_registry=None))
    assert runner._validate_args(dataclasses.replace(args, task_suite_name="libero_10", route_registry=None)) is None
    route = runner._validate_args(args)
    runner.verify_benchmark_bddl(route, route.bddl.path)
    other = tmp_path / "other.bddl"
    other.write_text("wrong actual benchmark task")
    with pytest.raises(ValueError, match="benchmark BDDL"):
        runner.verify_benchmark_bddl(route, other)
    with pytest.raises(ValueError, match="runner phase"):
        runner._validate_args(dataclasses.replace(args, phase_names="wrong"))


def test_rendered_schedule_binds_suite_without_changing_legacy_phase_names(tmp_path):
    path, entry = _registry(tmp_path)
    args = _runner_args(tmp_path, path, entry)
    payload = runner.render_libero_schedule(
        args.schedule, tmp_path / "rendered.json", task_id=0, phases=("approach", "grasp"),
        task_suite_name="libero_spatial", bddl_sha256=entry["bddl"]["sha256"],
    )
    assert payload["task_suite_name"] == "libero_spatial"
    assert payload["bddl_sha256"] == entry["bddl"]["sha256"]
    assert payload["phase_schedule"] == [["task_00:approach", 3], ["task_00:grasp", 1]]
    assert payload["maximum_stride"] == 3


def test_reserved_config_must_match_bound_suite_task(tmp_path):
    path, entry = _registry(tmp_path)
    args = _runner_args(tmp_path, path, entry)
    route = runner._validate_args(args)
    config = tmp_path / "config.json"
    payload = {"speed_schedule": json.loads(args.schedule.read_text()), "requested_rollouts": 5}
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task_identity"):
        runner.validate_search_run_config(config, schedule_path=args.schedule, num_trials=5, route=route)
    payload["task_identity"] = dataclasses.asdict(route.identity)
    config.write_text(json.dumps(payload))
    runner.validate_search_run_config(config, schedule_path=args.schedule, num_trials=5, route=route)
    payload["task_identity"]["suite"] = "libero_goal"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task_identity"):
        runner.validate_search_run_config(config, schedule_path=args.schedule, num_trials=5, route=route)


def test_runner_passes_registered_suite_to_evaluator_without_physics(tmp_path, monkeypatch):
    import sys
    import types

    path, entry = _registry(tmp_path)
    args = _runner_args(tmp_path, path, entry)
    route = runner._validate_args(args)
    (tmp_path / "config.json").write_text(json.dumps({
        "speed_schedule": json.loads(args.schedule.read_text()), "requested_rollouts": args.num_trials,
        "task_identity": dataclasses.asdict(route.identity),
    }))
    benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {
        "libero_spatial": lambda: types.SimpleNamespace(get_task=lambda _: types.SimpleNamespace(
            problem_folder="libero_spatial", bddl_file="task.bddl")),
    })
    fake_libero = types.ModuleType("libero.libero")
    fake_libero.benchmark = benchmark
    fake_libero.get_libero_path = lambda _: str(tmp_path)
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero)
    recorded = []
    main = types.ModuleType("examples.libero.speed_baselines.main")
    main.Args = types.SimpleNamespace
    main.eval_speed_baseline = recorded.append
    monkeypatch.setitem(sys.modules, main.__name__, main)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "mock-commit")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: object())
    monkeypatch.setattr(runner, "_port_is_open", lambda *a: False)
    monkeypatch.setattr(runner, "_wait_for_server", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_stop_process_group", lambda *a: None)
    runner.run(args)
    assert len(recorded) == 1
    assert recorded[0].task_suite_name == "libero_spatial"
    assert recorded[0].task_start == 0 and recorded[0].task_count == 1
    assert recorded[0].method == "strider" and recorded[0].fast_stride == 3


def test_baseline_and_challenger_cannot_bind_different_bddl_for_same_task(tmp_path):
    path, entry = _registry(tmp_path)
    other_bddl = tmp_path / "different.bddl"
    other_bddl.write_text("different physical goal")
    baseline = {"suite": entry["suite"], "task_id": 0, "method": "uniform", "uniform_stride": 2,
                "bddl": {"path": str(other_bddl), "sha256": routes.file_sha256(other_bddl)}}
    _save(path, [entry, baseline])
    with pytest.raises(ValueError, match="conflicting BDDL"):
        routes.load_routes(path)
