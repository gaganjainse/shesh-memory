"""Offline failure-memory loop tests.

The habit learner must treat failure observations honestly: repeated
failures must NOT promote a habit, reliability must drag confidence
down, and promoted habits that stop recurring (or start failing) must
decay toward archive. These are pure offline tests — no network, no
model — with deterministic time injection.

This covers the "failure-memory offline loop" gap: the learner's
failure path (success=False observations over a loop of turns) was
previously untested.
"""
from __future__ import annotations

import pytest

from shesh_memory.habits import (
    ARCHIVE_BELOW,
    PROMOTE_AT,
    Habit,
    HabitLearner,
)


class _FakeStore:
    """In-memory store stand-in with the same read/write surface."""

    def __init__(self) -> None:
        self.habits: dict = {}

    def read_habits(self) -> dict:
        return self.habits

    def write_habits(self, data: dict) -> None:
        self.habits = dict(data)


class _Clock:
    """Injectable clock so decay tests do not sleep for weeks."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Deterministic time: patches habits.time.time for observe/decay."""
    c = _Clock()
    monkeypatch.setattr("shesh_memory.habits.time.time", c.now)
    return c


def _learner() -> HabitLearner:
    return HabitLearner(_FakeStore())


def _age_all(learner: HabitLearner, seconds: float, clock: _Clock) -> None:
    """Move both last_seen and last_decayed into the past (fake store) so
    tick_decay sees the habit as old under the time-based decay model."""
    for h in learner.habits.values():
        h.last_seen = clock.t - seconds
        h.last_decayed = clock.t - seconds


def test_repeated_failures_never_promote(clock: _Clock) -> None:
    """A habit that always fails must stay below the promotion threshold."""
    learner = _learner()
    h: Habit | None = None
    for _ in range(50):
        h = learner.observe("action:deploy|hour:3", "deploys at 3am", success=False)
        clock.advance(60)
    assert h is not None
    assert h.count == 51  # 50 observes + the creating one
    assert h.successes == 0
    assert not h.promoted
    # confidence is reliability-weighted: 0% success -> 0 confidence
    assert h.confidence == 0.0


def test_high_volume_failures_still_never_promote(clock: _Clock) -> None:
    """Regression: confidence alone is volume-weighted and would cross the
    threshold at ~150 observations even with 0% success. The reliability
    floor must refuse promotion no matter the volume."""
    learner = _learner()
    h: Habit | None = None
    for _ in range(300):
        h = learner.observe("action:flaky|hour:9", "flaky routine", success=False)
        clock.advance(60)
    assert h is not None
    assert h.count == 301
    assert h.successes == 0
    # volume alone used to push confidence past PROMOTE_AT; with the
    # reliability-weighted formula it stays exactly 0
    assert h.confidence == 0.0
    assert not h.promoted


def test_failure_loop_drags_promoted_habit_down(clock: _Clock) -> None:
    """A promoted habit that starts failing loses confidence over the loop."""
    learner = _learner()
    h: Habit | None = None
    # 30 successes -> promoted
    for _ in range(30):
        h = learner.observe("action:backup|hour:4", "backs up at 4am", success=True)
        clock.advance(60)
    assert h is not None and h.promoted
    peak = h.confidence

    # 60 failures -> reliability collapses, confidence drags toward 0.
    # Promotion is sticky (once proposed, never silently demoted), so we
    # assert the collapse, not demotion.
    for _ in range(60):
        h = learner.observe("action:backup|hour:4", "backs up at 4am", success=False)
        clock.advance(60)
    assert h.confidence < peak
    assert h.successes == 30
    assert h.count == 91  # 90 observes + the creating one
    assert h.confidence < peak / 2


def test_stale_habit_decays_and_archives(clock: _Clock) -> None:
    """A promoted habit not seen for weeks decays below floor and archives."""
    learner = _learner()
    h: Habit | None = None
    for _ in range(30):
        h = learner.observe("action:focus|hour:10", "focus at 10am", success=True)
        clock.advance(60)
    assert h is not None and h.promoted and not h.archived
    _age_all(learner, 0, clock)  # base timestamps on the fake clock

    # 8 weeks of silence (4 half-lives) -> confidence * (0.5**4) = /16
    clock.advance(8 * 7 * 24 * 3600)
    archived = learner.tick_decay()
    assert h.archived
    assert h in archived
    assert h.confidence < ARCHIVE_BELOW
    assert h not in list(learner.active_habits())


def test_failing_then_recovering_loop(clock: _Clock) -> None:
    """Recovering behavior re-raises confidence but needs real evidence."""
    learner = _learner()
    h: Habit | None = None
    for _ in range(10):
        h = learner.observe("action:focus|hour:10", "focus at 10am", success=False)
        clock.advance(60)
    assert h is not None and not h.promoted
    # 40 consecutive wins after the failures
    for _ in range(40):
        h = learner.observe("action:focus|hour:10", "focus at 10am", success=True)
        clock.advance(60)
    assert h.promoted
    assert h.successes == 40
    assert h.count == 51  # 50 observes + the creating one
    assert h.confidence >= PROMOTE_AT


def test_decay_advances_with_age_not_tick_count(clock: _Clock) -> None:
    """Decay is driven by age since last seen, not by how many ticks ran."""
    learner = _learner()
    h: Habit | None = None
    for _ in range(30):
        h = learner.observe("action:meditate|hour:6", "meditates at 6am", success=True)
        clock.advance(60)
    assert h is not None and h.promoted
    _age_all(learner, 0, clock)  # base timestamps on the fake clock
    clock.advance(30 * 24 * 3600)  # ~1 month
    learner.tick_decay()
    c1 = h.confidence
    # a second tick with no time passing must not double-penalize
    learner.tick_decay()
    assert h.confidence == c1
    # but real age keeps decaying it
    clock.advance(14 * 24 * 3600)  # one more half-life
    learner.tick_decay()
    assert h.confidence < c1


def test_archive_only_hits_promoted_habits(clock: _Clock) -> None:
    """Unpromoted (candidate) habits are not archived by decay."""
    learner = _learner()
    h = learner.observe("action:x|hour:1", "x", success=True)
    assert not h.promoted
    _age_all(learner, 0, clock)  # base timestamps on the fake clock
    clock.advance(60 * 24 * 3600)  # 60 days
    archived = learner.tick_decay()
    assert archived == []
    assert not h.archived
