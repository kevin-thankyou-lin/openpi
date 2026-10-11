"""Durable per-task experiment accounting. Record a start *before* stepping physics.

The caller owns worker lifecycle: only reconcile after proving its worker stopped.
A crash after a start receipt but before physics conservatively consumes that slot.
No cap interpretation, category exemption, or budget reset is implicit.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path

PURPOSES = frozenset({"baseline", "search_candidate", "causal_replay", "candidate_validation"})


class BudgetError(RuntimeError):
    """Refuse execution when accounting is unknown, inconsistent, or exhausted."""


@dataclass(frozen=True)
class Cap:
    unit: str
    limit: int
    purposes: tuple[str, ...]

    def __post_init__(self):
        if self.unit not in {"episodes", "candidates"} or type(self.limit) is not int or self.limit < 0:
            raise ValueError("Explicit nonnegative episode or candidate cap required")
        if not self.purposes or not set(self.purposes) <= PURPOSES or len(set(self.purposes)) != len(self.purposes):
            raise ValueError("Explicit unique included purposes required")

    def record(self):
        return {"unit": self.unit, "limit": self.limit, "purposes": sorted(self.purposes)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pins(value):
    if (
        not isinstance(value, dict)
        or not value
        or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in value.items())
    ):
        raise ValueError("Nonempty named source contract pins required")
    return dict(value)


class CampaignBudget:
    """One immutable campaign/task identity; all purposes remain visible in totals.

    ``candidate_id`` must identify the complete scientific controller, including
    composition, gates, policy and reset convention, rather than its speed table alone.
    Imported historical starts always count, even when they exceed the configured cap.
    Such imports block further reservations rather than hiding prior spending.
    """

    def __init__(self, path, *, campaign_id, suite, task, bddl_sha256, source_pins):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not all(isinstance(x, str) and x for x in [campaign_id, suite, bddl_sha256]):
            raise ValueError("Explicit campaign, suite and BDDL identity required")
        if (
            type(task) is not int
            or task < 0
            or len(bddl_sha256) != 64
            or any(c not in "0123456789abcdef" for c in bddl_sha256)
        ):
            raise ValueError("Invalid task or BDDL SHA256")
        self.identity = {"campaign_id": campaign_id, "suite": suite, "task": task, "bddl_sha256": bddl_sha256}
        self.source_pins = _pins(source_pins)
        with self._locked() as events:
            metadata = {"kind": "metadata", "schema": 1, "identity": self.identity, "source_pins": self.source_pins}
            if not events:
                self._append(events, metadata)
            elif events[0]["event"] != metadata:
                raise BudgetError("Ledger identity or source contract changed; do not reset accounting")

    @contextmanager
    def _locked(self):
        # A separate never-replaced inode keeps locking correct across journal appends.
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            events = []
            raw = self.path.read_bytes() if self.path.exists() else b""
            if raw and not raw.endswith(b"\n"):
                raise BudgetError("Incomplete journal tail; preserve it and recover explicitly")
            previous = None
            for index, line in enumerate(raw.splitlines()):
                try:
                    row = json.loads(line)
                    unsigned = {k: row[k] for k in ["sequence", "previous", "event"]}
                    valid = (
                        row["sequence"] == index and row["previous"] == previous and row["sha256"] == _digest(unsigned)
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    raise BudgetError("Invalid journal record") from exc
                if not valid:
                    raise BudgetError("Broken journal hash chain")
                events.append(row)
                previous = row["sha256"]
            yield events

    def _append(self, events, event):
        row = {"sequence": len(events), "previous": events[-1]["sha256"] if events else None, "event": event}
        row["sha256"] = _digest(row)
        data = (_canonical(row) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        events.append(row)

    @staticmethod
    def _state(events):
        state = {"cap": None, "caps": {}, "reservations": {}, "starts": {}, "imports": {}, "outcomes": {}}
        for row in events:
            e = row["event"]
            if e["kind"] == "cap":
                state["caps"][e["name"]] = e
                if e["name"] == "primary":
                    state["cap"] = e
            elif e["kind"] == "reserve":
                state["reservations"][e["id"]] = dict(e, closed=False)
            elif e["kind"] == "start":
                state["starts"][e["id"]] = e
            elif e["kind"] == "import":
                state["imports"][e["id"]] = e
            elif e["kind"] == "outcome":
                state["outcomes"][e["id"]] = e["outcome"]
            elif e["kind"] == "close":
                state["reservations"][e["id"]]["closed"] = True
        return state

    @staticmethod
    def _usage(state, selected_cap=None):
        physical = list(state["starts"].values()) + list(state["imports"].values())
        totals = {purpose: sum(x["purpose"] == purpose for x in physical) for purpose in sorted(PURPOSES)}
        if selected_cap is None:
            selected_cap = state["cap"]
        if selected_cap is None:
            return {"by_purpose": totals, "total_starts": len(physical), "cap": None, "spent": None, "reserved": None}
        cap = selected_cap["cap"]
        selected = [x for x in physical if x["purpose"] in cap["purposes"]]
        active = [r for r in state["reservations"].values() if not r["closed"] and r["purpose"] in cap["purposes"]]
        if cap["unit"] == "episodes":
            spent = len(selected)
            reserved = sum(
                r["max_starts"] - sum(s["reservation"] == r["id"] for s in state["starts"].values()) for r in active
            )
        else:
            tested = {x["candidate"] for x in selected}
            spent = len(tested)
            reserved = len({r["candidate"] for r in active} - tested)
        return {"by_purpose": totals, "total_starts": len(physical), "cap": cap, "spent": spent, "reserved": reserved}

    @classmethod
    def _check_limits(cls, state, purpose):
        if not state["caps"] or not any(purpose in c["cap"]["purposes"] for c in state["caps"].values()):
            raise BudgetError("Cap unit, scope and included purposes unresolved for this purpose")
        for cap in state["caps"].values():
            usage = cls._usage(state, cap)
            if usage["spent"] + usage["reserved"] > usage["cap"]["limit"]:
                raise BudgetError("Budget exhausted or already reserved (possibly by historical imports)")

    def snapshot(self):
        with self._locked() as events:
            state = self._state(events)
            return dict(
                identity=self.identity,
                source_pins=self.source_pins,
                **self._usage(state),
                limits={name: self._usage(state, cap) for name, cap in state["caps"].items()},
            )

    def configure(self, cap: Cap, *, authority, name="primary"):
        """Seal one explicit interpretation. Changing it requires a reviewed ledger migration."""
        if not isinstance(name, str) or not name or not isinstance(authority, str) or not authority:
            raise ValueError("Cap authority/provenance required")
        event = {"kind": "cap", "name": name, "cap": cap.record(), "authority": authority}
        with self._locked() as events:
            previous = self._state(events)["caps"].get(name)
            if previous and previous != event:
                raise BudgetError("Cap already sealed")
            if not previous:
                self._append(events, event)

    @staticmethod
    def _fields(identifier, candidate, purpose, episode=None):
        if not isinstance(identifier, str) or not identifier or not isinstance(candidate, str) or not candidate:
            raise ValueError("Stable nonempty event and candidate IDs required")
        if purpose not in PURPOSES or (episode is not None and (type(episode) is not int or episode < 0)):
            raise ValueError("Invalid purpose or episode")

    def reserve(self, reservation_id, *, candidate_id, purpose, max_starts):
        self._fields(reservation_id, candidate_id, purpose)
        if type(max_starts) is not int or max_starts < 1:
            raise ValueError("Positive explicit reservation size required")
        event = {
            "kind": "reserve",
            "id": reservation_id,
            "candidate": candidate_id,
            "purpose": purpose,
            "max_starts": max_starts,
        }
        with self._locked() as events:
            state = self._state(events)
            previous = state["reservations"].get(reservation_id)
            if previous:
                if {k: previous[k] for k in event} != event:
                    raise BudgetError("Reservation ID reused with different content")
                return
            state["reservations"][reservation_id] = dict(event, closed=False)
            self._check_limits(state, purpose)
            self._append(events, event)

    def start(self, reservation_id, *, start_id, episode):
        """Consume before physics; return True only for a new start.

        A False return means already recorded: do NOT execute that start again.
        """
        if type(episode) is not int or episode < 0:
            raise ValueError("Explicit nonnegative episode required")
        with self._locked() as events:
            state = self._state(events)
            r = state["reservations"].get(reservation_id)
            if r is None:
                raise BudgetError("Unknown reservation")
            self._fields(start_id, r["candidate"], r["purpose"], episode)
            event = {
                "kind": "start",
                "id": start_id,
                "reservation": reservation_id,
                "candidate": r["candidate"],
                "purpose": r["purpose"],
                "episode": episode,
            }
            if start_id in state["imports"]:
                raise BudgetError("Start ID already imported; do not execute or count it again")
            previous = state["starts"].get(start_id)
            if previous:
                if previous != event:
                    raise BudgetError("Start ID reused with different content")
                return False
            if (
                r["closed"]
                or sum(x["reservation"] == reservation_id for x in state["starts"].values()) >= r["max_starts"]
            ):
                raise BudgetError("Reservation closed or exhausted")
            self._check_limits(state, r["purpose"])
            self._append(events, event)
            return True

    def outcome(self, start_id, outcome):
        if outcome not in {"success", "failure", "infrastructure_failure"}:
            raise ValueError("Explicit terminal outcome required")
        with self._locked() as events:
            state = self._state(events)
            if start_id not in state["starts"]:
                raise BudgetError("Unknown physical start")
            previous = state["outcomes"].get(start_id)
            if previous and previous != outcome:
                raise BudgetError("Outcome already sealed")
            if not previous:
                self._append(events, {"kind": "outcome", "id": start_id, "outcome": outcome})

    def reconcile(self, reservation_id, *, worker_stopped):
        """Close an abandoned/finished worker; only unused slots are released.

        Unfinished starts become infrastructure failures and remain counted. Never
        call with a live worker; lifecycle verification belongs to the dispatcher.
        """
        if worker_stopped is not True:
            raise BudgetError("Worker must be proved stopped before reconciliation")
        with self._locked() as events:
            state = self._state(events)
            r = state["reservations"].get(reservation_id)
            if r is None:
                raise BudgetError("Unknown reservation")
            if r["closed"]:
                return
            for s in state["starts"].values():
                if s["reservation"] == reservation_id and s["id"] not in state["outcomes"]:
                    self._append(events, {"kind": "outcome", "id": s["id"], "outcome": "infrastructure_failure"})
            self._append(events, {"kind": "close", "id": reservation_id})

    def import_start(self, source_event_id, *, candidate_id, purpose, episode, task_identity, source_pins):
        """Import one historical physical start, keyed by original event identity, not mirror path."""
        if type(episode) is not int or episode < 0:
            raise ValueError("Explicit nonnegative episode required")
        self._fields(source_event_id, candidate_id, purpose, episode)
        expected = {k: self.identity[k] for k in ["suite", "task", "bddl_sha256"]}
        if task_identity != expected:
            raise BudgetError("Historical task identity mismatch")
        event = {
            "kind": "import",
            "id": source_event_id,
            "candidate": candidate_id,
            "purpose": purpose,
            "episode": episode,
            "task_identity": task_identity,
            "source_pins": _pins(source_pins),
        }
        with self._locked() as events:
            state = self._state(events)
            if source_event_id in state["starts"]:
                raise BudgetError("Historical event is already a local start; do not count it again")
            previous = state["imports"].get(source_event_id)
            if previous and previous != event:
                raise BudgetError("Historical event identity reused with different content")
            if not previous:
                self._append(events, event)
