"""Habit and intention learning.

This is not model training. It is frequency + recency + corroboration counting
over structured observations, promoting repeated patterns into "habits" and
decaying ones that stop recurring (discard the dross).

A habit is something like:
  - "starts focus mode around 10:00 on weekdays"
  - "prefers summaries in bullet points"
  - "commits Rust projects before running tests"

Observations come from actions Shesha takes/sees. The learner:
  1. normalizes them into a signature,
  2. counts occurrences and tracks first/last seen and success rate,
  3. promotes to a learned habit when evidence crosses a threshold,
  4. decays confidence when not seen,
  5. archives habits whose confidence falls below a floor.

Promotions are *candidates* the user/agent reviews; we never silently change
behavior based on one coincidence.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

# A signature is a short stable string, e.g. "action:focus|dow:1|hour:10".


@dataclass
class Habit:
    signature: str
    description: str
    count: int = 1
    successes: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.0
    promoted: bool = False
    archived: bool = False

    def observe(self, success: bool, now: float) -> None:
        self.count += 1
        self.last_seen = now
        if success:
            self.successes += 1
        # confidence rises with corroboration, capped; reward reliability.
        reliability = self.successes / self.count
        self.confidence = min(1.0, 0.2 * math.log1p(self.count) * (0.5 + 0.5 * reliability))


PROMOTE_AT = 0.5      # confidence needed to propose a habit
ARCHIVE_BELOW = 0.15  # confidence after decay at which to archive
HALF_LIFE_S = 14 * 24 * 3600  # ~2 weeks


def decay(habit: Habit, now: float) -> None:
    """Exponentially decay confidence for habits not recently observed."""
    age = max(0.0, now - habit.last_seen)
    factor = 0.5 ** (age / HALF_LIFE_S)
    habit.confidence *= factor
    if habit.confidence < ARCHIVE_BELOW and habit.promoted:
        habit.archived = True


class HabitLearner:
    def __init__(self, store) -> None:  # store: MemoryStore
        self.store = store
        self.habits: dict[str, Habit] = {
            sig: Habit(**data) for sig, data in store.read_habits().items()
        }

    def observe(self, signature: str, description: str, *, success: bool = True) -> Habit:
        now = time.time()
        h = self.habits.get(signature)
        if h is None:
            h = Habit(signature=signature, description=description)
            self.habits[signature] = h
        h.observe(success, now)
        if not h.promoted and h.confidence >= PROMOTE_AT:
            h.promoted = True   # candidate for review
        self._persist()
        return h

    def active_habits(self) -> Iterable[Habit]:
        return [h for h in self.habits.values() if h.promoted and not h.archived]

    def tick_decay(self) -> list[Habit]:
        """Call periodically (e.g., daily). Returns newly archived habits."""
        now = time.time()
        archived = []
        for h in self.habits.values():
            before = h.archived
            decay(h, now)
            if not before and h.archived:
                archived.append(h)
        self._persist()
        return archived

    def _persist(self) -> None:
        self.store.write_habits({
            sig: {
                "signature": h.signature, "description": h.description,
                "count": h.count, "successes": h.successes,
                "first_seen": h.first_seen, "last_seen": h.last_seen,
                "confidence": round(h.confidence, 3),
                "promoted": h.promoted, "archived": h.archived,
            }
            for sig, h in self.habits.items()
        })
