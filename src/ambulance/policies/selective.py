"""Selective dispatch: is it ever right to hold an ambulance back?

Every policy so far sends the nearest available unit to whatever call arrives. That is
myopically optimal per call and ignores urgency: committing the last free unit to a
routine call leaves nothing for a cardiac arrest two minutes later. Real services do
withhold, and this makes the question measurable -- the reserve threshold is swept and the
cost is read off the two priority classes separately.

The withholding is a *delay*, not a refusal. A held call goes to the queue and is given
the next unit that frees, ahead of nothing and behind any higher-priority call. So the
trade being measured is response time for routine calls against readiness for urgent ones,
never a call dropped.
"""

from __future__ import annotations

from ..city import Call
from .base import Policy
from .coverage import DMEXCLP


class SelectiveDispatch(Policy):
    """Wraps a policy and withholds units from low-priority calls when cover is thin.

    A call of priority worse than `protected_priority` is queued rather than dispatched
    whenever at most `reserve` units are available, keeping those units free for an urgent
    call that may be seconds away.
    """

    def __init__(
        self,
        inner: Policy | None = None,
        reserve: int = 1,
        protected_priority: int = 1,
    ) -> None:
        if reserve < 0:
            raise ValueError(f"reserve must be non-negative, got {reserve}")
        self.inner = inner if inner is not None else DMEXCLP()
        self.reserve = reserve
        self.protected_priority = protected_priority

    @property
    def name(self) -> str:
        return f"selective(reserve={self.reserve})"

    def reset(self) -> None:
        self.inner.reset()

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None

        if call.priority > self.protected_priority and len(available) <= self.reserve:
            # Only withhold while something is actually out on a job. With the whole
            # fleet idle nothing would ever become free, no BECOME_FREE event would fire,
            # and the held call would wait for the rest of the simulation -- a deadlock
            # rather than a trade-off. Tested, not assumed.
            if len(available) < len(sim.ambulances):
                return None

        return self.inner.dispatch(sim, call)

    def reposition(self, sim, amb_id: int) -> int:
        return self.inner.reposition(sim, amb_id)
