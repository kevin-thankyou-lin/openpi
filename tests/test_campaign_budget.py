"""CPU-only accounting tests; no policy, simulator or GPU imports."""

import concurrent.futures
import json
import multiprocessing
import os

import pytest

from examples.libero.speed_baselines.campaign_budget import BudgetError
from examples.libero.speed_baselines.campaign_budget import CampaignBudget
from examples.libero.speed_baselines.campaign_budget import Cap

IDENTITY = {"campaign_id": "search-v1", "suite": "libero_10", "task": 3, "bddl_sha256": "a" * 64}
PINS = {"policy": "frozen-policy", "evaluator": "frozen-evaluator", "reset": "official-v1"}
TASK = {k: IDENTITY[k] for k in ["suite", "task", "bddl_sha256"]}


def ledger(path):
    return CampaignBudget(path, **IDENTITY, source_pins=PINS)


def configured(path, limit=2, unit="episodes", purposes=("search_candidate", "causal_replay")):
    budget = ledger(path)
    budget.configure(Cap(unit, limit, purposes), authority="explicit test contract")
    return budget


def attempt_reservation(args):
    path, number = args
    budget = ledger(path)
    try:
        budget.reserve(f"worker-{number}", candidate_id=f"candidate-{number}", purpose="search_candidate", max_starts=1)
        return True
    except BudgetError:
        return False


def crash_after_start(path):
    budget = ledger(path)
    assert budget.start("worker", start_id="attempt/episode0", episode=0)
    os._exit(17)


def test_unknown_cap_and_immutable_identity(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = ledger(path)
    with pytest.raises(BudgetError, match="unresolved"):
        budget.reserve("worker", candidate_id="candidate", purpose="baseline", max_starts=1)
    with pytest.raises(BudgetError, match="identity or source"):
        CampaignBudget(path, **IDENTITY, source_pins={"policy": "different"})
    budget.configure(Cap("episodes", 2, ("search_candidate",)), authority="user clarification")
    budget.configure(Cap("episodes", 2, ("search_candidate",)), authority="user clarification")
    with pytest.raises(BudgetError, match="sealed"):
        budget.configure(Cap("episodes", 20, ("search_candidate",)), authority="silent reset")


def test_failures_count_and_only_unused_reservations_refund(tmp_path):
    budget = configured(tmp_path / "budget.jsonl", limit=3)
    budget.reserve("worker", candidate_id="controller-A", purpose="search_candidate", max_starts=3)
    assert budget.start("worker", start_id="start0", episode=0)
    assert budget.start("worker", start_id="start0", episode=0) is False
    budget.outcome("start0", "failure")
    assert budget.snapshot()["spent"] == 1
    assert budget.snapshot()["reserved"] == 2
    with pytest.raises(BudgetError):
        budget.reserve("extra", candidate_id="controller-A", purpose="search_candidate", max_starts=1)
    budget.reconcile("worker", worker_stopped=True)
    budget.reconcile("worker", worker_stopped=True)
    assert budget.snapshot()["spent"] == 1
    assert budget.snapshot()["reserved"] == 0
    budget.reserve("next", candidate_id="controller-A", purpose="causal_replay", max_starts=2)
    with pytest.raises(BudgetError, match="closed"):
        budget.start("worker", start_id="late", episode=1)


def test_concurrent_atomic_reservations_never_overrun(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = configured(path, limit=3)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt_reservation, [(path, i) for i in range(16)]))
    assert sum(results) == 3
    assert budget.snapshot()["reserved"] == 3


def test_real_process_crash_keeps_consumed_start(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = configured(path, limit=2)
    budget.reserve("worker", candidate_id="A", purpose="search_candidate", max_starts=2)
    process = multiprocessing.Process(target=crash_after_start, args=(path,))
    process.start()
    process.join(10)
    assert process.exitcode == 17
    recovered = ledger(path)
    assert recovered.snapshot()["spent"] == 1
    assert recovered.snapshot()["reserved"] == 1
    with pytest.raises(BudgetError, match="stopped"):
        recovered.reconcile("worker", worker_stopped=False)
    recovered.reconcile("worker", worker_stopped=True)
    assert recovered.snapshot()["spent"] == 1
    assert recovered.snapshot()["reserved"] == 0
    with pytest.raises(BudgetError, match="sealed"):
        recovered.outcome("attempt/episode0", "success")
    recovered.reserve("replacement", candidate_id="B", purpose="causal_replay", max_starts=1)


def test_import_idempotence_and_historical_overrun_are_visible(tmp_path):
    budget = configured(tmp_path / "budget.jsonl", limit=1)
    receipt = {
        "candidate_id": "old-A",
        "purpose": "search_candidate",
        "episode": 7,
        "task_identity": TASK,
        "source_pins": PINS,
    }
    budget.import_start("old-attempt/7", **receipt)
    before = budget.path.read_bytes()
    budget.import_start("old-attempt/7", **receipt)
    assert before == budget.path.read_bytes()
    with pytest.raises(BudgetError, match="different content"):
        budget.import_start("old-attempt/7", **dict(receipt, episode=8))
    budget.import_start("old-attempt/8", **dict(receipt, episode=8))
    assert budget.snapshot()["spent"] == 2
    with pytest.raises(BudgetError, match="exhausted"):
        budget.reserve("new", candidate_id="B", purpose="search_candidate", max_starts=1)
    with pytest.raises(BudgetError, match="task identity"):
        budget.import_start("wrong-task", **dict(receipt, task_identity=dict(TASK, task=4)))


def test_import_after_reservation_blocks_start_if_cap_exceeded(tmp_path):
    budget = configured(tmp_path / "budget.jsonl", limit=1)
    budget.reserve("new", candidate_id="B", purpose="search_candidate", max_starts=1)
    budget.import_start(
        "old-start", candidate_id="A", purpose="causal_replay", episode=1, task_identity=TASK, source_pins=PINS
    )
    with pytest.raises(BudgetError, match="historical imports"):
        budget.start("new", start_id="new-start", episode=2)


def test_candidate_count_is_distinct_from_episode_count(tmp_path):
    budget = configured(tmp_path / "budget.jsonl", limit=1, unit="candidates")
    budget.reserve("one", candidate_id="A", purpose="search_candidate", max_starts=2)
    budget.reserve("same-controller", candidate_id="A", purpose="causal_replay", max_starts=1)
    assert budget.snapshot()["reserved"] == 1
    with pytest.raises(BudgetError):
        budget.reserve("different", candidate_id="B", purpose="search_candidate", max_starts=1)
    for i in range(2):
        budget.start("one", start_id=f"start-{i}", episode=i)
    budget.start("same-controller", start_id="replay-0", episode=0)
    assert budget.snapshot()["spent"] == 1
    assert budget.snapshot()["total_starts"] == 3
    with pytest.raises(BudgetError, match="exhausted"):
        budget.start("one", start_id="extra", episode=3)


def test_excluded_purposes_stay_visible(tmp_path):
    budget = configured(tmp_path / "budget.jsonl", limit=0)
    with pytest.raises(BudgetError, match="unresolved"):
        budget.reserve("unlimited-baseline", candidate_id="A", purpose="baseline", max_starts=1)
    budget.configure(
        Cap("episodes", 2, ("baseline", "candidate_validation")), authority="explicit extra limit", name="extra"
    )
    for purpose in ["baseline", "candidate_validation"]:
        budget.reserve(purpose, candidate_id="A", purpose=purpose, max_starts=1)
        budget.start(purpose, start_id=purpose, episode=0)
    assert budget.snapshot()["spent"] == 0
    assert budget.snapshot()["by_purpose"] == {
        "baseline": 1,
        "causal_replay": 0,
        "search_candidate": 0,
        "candidate_validation": 1,
    }


def test_corrupt_or_partial_journal_fails_closed(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger(path)
    with path.open("ab") as stream:
        stream.write(b'{"unfinished":')
    with pytest.raises(BudgetError, match="Incomplete"):
        ledger(path)
    path2 = tmp_path / "tampered.jsonl"
    ledger(path2)
    row = json.loads(path2.read_text())
    row["event"]["identity"]["task"] = 9
    path2.write_text(json.dumps(row) + "\n")
    with pytest.raises(BudgetError, match="hash chain"):
        ledger(path2)


def test_confirmed_twenty_challenger_episodes_include_replays_and_validation(tmp_path):
    budget = configured(
        tmp_path / "budget.jsonl", limit=20, purposes=("search_candidate", "causal_replay", "candidate_validation")
    )
    # Existing current campaign work cannot be silently reset by a new ledger consumer.
    for i in range(18):
        budget.import_start(
            f"original-campaign/{i}",
            candidate_id="old-controller",
            purpose="search_candidate",
            episode=i,
            task_identity=TASK,
            source_pins=PINS,
        )
    for purpose in ["causal_replay", "candidate_validation"]:
        budget.reserve(purpose, candidate_id="new-controller", purpose=purpose, max_starts=1)
        budget.start(purpose, start_id=purpose, episode=18)
        budget.outcome(purpose, "failure")
    assert budget.snapshot()["spent"] == 20
    with pytest.raises(BudgetError, match="exhausted"):
        budget.reserve("twenty-first", candidate_id="new-controller", purpose="search_candidate", max_starts=1)
    # Baselines require their own explicit bound and do not consume the challenger cap.
    budget.configure(Cap("episodes", 2, ("baseline",)), authority="two registered baseline controls", name="baselines")
    budget.reserve("ds2", candidate_id="DS2", purpose="baseline", max_starts=2)
    assert budget.snapshot()["limits"]["baselines"]["reserved"] == 2
    with pytest.raises(BudgetError, match="exhausted"):
        budget.reserve("unbounded-baseline", candidate_id="DS1", purpose="baseline", max_starts=1)
    budget.start("ds2", start_id="ds2/0", episode=0)
    assert budget.snapshot()["spent"] == 20
    assert budget.snapshot()["total_starts"] == 21


def attempt_start(args):
    path, number = args
    try:
        return ledger(path).start("shared", start_id=f"concurrent-start-{number}", episode=number)
    except BudgetError:
        return False


def test_concurrent_starts_cannot_exceed_shared_reservation(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = configured(path, limit=2)
    budget.reserve("shared", candidate_id="A", purpose="search_candidate", max_starts=2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        starts = list(pool.map(attempt_start, [(path, i) for i in range(12)]))
    assert sum(starts) == 2
    assert budget.snapshot()["spent"] == 2
    assert budget.snapshot()["reserved"] == 0


def test_import_cannot_double_count_local_start_identity(tmp_path):
    budget = configured(tmp_path / "budget.jsonl")
    budget.reserve("run", candidate_id="A", purpose="search_candidate", max_starts=2)
    budget.start("run", start_id="original-id", episode=0)
    with pytest.raises(BudgetError, match="already a local start"):
        budget.import_start(
            "original-id", candidate_id="A", purpose="search_candidate", episode=0, task_identity=TASK, source_pins=PINS
        )
    budget.import_start(
        "historical-id", candidate_id="A", purpose="search_candidate", episode=1, task_identity=TASK, source_pins=PINS
    )
    with pytest.raises(BudgetError, match="already imported"):
        budget.start("run", start_id="historical-id", episode=1)
