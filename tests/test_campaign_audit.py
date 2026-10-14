import copy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero.speed_baselines.actions import SupActionComposer
from examples.libero.speed_baselines.actions import uniform_slices
from examples.libero.speed_baselines.campaign_audit import audit_episode
from examples.libero.speed_baselines.campaign_policy import array_hash


@pytest.fixture
def evidence():
    route = SimpleNamespace(
        method="uniform",
        uniform_stride=2,
        identity=SimpleNamespace(suite="libero_spatial", task_id=0, bddl_sha256="a" * 64),
    )
    composer = SupActionComposer(
        input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6), output_max=np.ones(6)
    )
    actions = np.arange(70, dtype=np.float64).reshape(10, 7) / 1000
    receipt = {
        "episode": 7,
        "plan": 0,
        "seed": 0,
        "rng_before": [0, 0],
        "rng_after": [1, 2],
        "suite": "libero_spatial",
        "task": 0,
        "bddl_sha256": "a" * 64,
        "raw_actions": actions.tolist(),
        "actions_dtype": "float64",
        "actions_sha256": array_hash(actions),
        "request_sha256": "b" * 64,
    }
    rows = [
        {"kind": "scene"},
        {"kind": "initial", "observations": {"camera": "pin"}},
        {"kind": "plan", "decision": {"decision_index": 0}, "receipt": receipt},
    ]
    for index, group in enumerate(uniform_slices(actions, stride=2, composer=composer)[:2]):
        rows.append(
            {
                "kind": "step",
                "env_step": index,
                "decision_index": 0,
                "source_indices": list(group.source_indices),
                "source_stride": group.stride,
                "action": group.action.tolist(),
            }
        )
    rows.append(
        {
            "kind": "terminal",
            "env_steps": 2,
            "success": True,
            "loop_wall_seconds": 1.5,
            "state": {"goal_satisfied": True, "goal_predicates": {"on bowl plate": True}},
        }
    )
    for row in rows:
        row["episode"] = 7
    result = {
        "episode_index": 7,
        "env_steps": 2,
        "success": True,
        "loop_wall_seconds": 1.5,
        "decision_count": 1,
        "source_actions_consumed": 4,
    }
    return rows, result, route, composer


def test_success_may_end_mid_chunk_without_being_a_discarded_replan(evidence):
    rows, result, route, composer = evidence
    ref = {"observations": rows[1]["observations"], "first_receipt": rows[2]["receipt"]}
    assert audit_episode(rows, result, route=route, composer=composer, expected_reference=ref)["passed"]


@pytest.mark.parametrize("fault", ["action", "duplicate_source", "rng", "rng_after", "phase", "initial", "truncated_failure"])
def test_rejects_corrupt_or_nonmatching_rollouts(evidence, fault):
    rows, result, route, composer = copy.deepcopy(evidence)
    ref = {"observations": rows[1]["observations"], "first_receipt": copy.deepcopy(rows[2]["receipt"])}
    if fault == "action":
        rows[3]["action"][0] += 0.001
    elif fault == "duplicate_source":
        rows[4]["source_indices"] = [0, 1]
    elif fault == "rng":
        rows[2]["receipt"]["rng_before"] = [0, 3]
    elif fault == "rng_after":
        ref["first_receipt"]["rng_after"] = [9, 9]
    elif fault == "phase":
        result["phase_decisions"] = [{}]
    elif fault == "initial":
        ref["first_receipt"]["actions_sha256"] = "c" * 64
    else:
        result["success"] = False
        rows[-1].update(success=False, state={"goal_satisfied": False, "goal_predicates": {"on bowl plate": False}})
    with pytest.raises(ValueError, match="mismatch|skip|reuse|selector|differs|early"):
        audit_episode(rows, result, route=route, composer=composer, expected_reference=ref)


def test_refresh_audit_reconstructs_changed_suffix_and_rejects_wrong_offset(evidence):
    rows,result,route,composer=copy.deepcopy(evidence)
    route.method='strider'
    raw=np.asarray(rows[2]['receipt']['raw_actions'])
    rows=rows[:3]
    rows[2]['env_step']=0
    rows[2]['decision']['speed_factors']=[1]*10
    refresh={'env_step':5,'policy_decision_index':0,
             'decision':{'source_offset':5,'speed_factors':[2]*5}}
    groups=[[i] for i in range(5)]+[[5,6],[7,8],[9]]
    for t,idx in enumerate(groups):
        if t==5:rows.append({'kind':'predictor_refresh','episode':7,**refresh})
        group=raw[idx]
        action=group[0] if len(idx)==1 else composer.merge(group)
        rows.append({'kind':'step','episode':7,'env_step':t,'decision_index':0,
                     'source_indices':idx,'source_stride':len(idx),'action':action.tolist()})
    rows.append({'kind':'terminal','episode':7,'env_steps':8,'success':True,
                 'loop_wall_seconds':1.5,'state':{'goal_satisfied':True,'goal_predicates':{'goal':True}}})
    result.update(env_steps=8,source_actions_consumed=10,phase_decisions=[{}]*8,
                  predictor_refreshes=[refresh],predictor_call_count=2,strider_min_prediction_hz=4)
    assert audit_episode(rows,result,route=route,composer=composer)['passed']
    refresh['decision']['source_offset']=4
    with pytest.raises(ValueError,match='offset mismatch'):
        audit_episode(rows,result,route=route,composer=composer)
