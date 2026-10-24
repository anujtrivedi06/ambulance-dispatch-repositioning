"""Policy interface shared by every method in the project."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..city import Call


class Policy(ABC):
    """A dispatch + repositioning policy.

    Both decisions go through one object so that heuristics, online search and learned
    policies all plug into the same evaluation harness unchanged.
    """

    name: str = "policy"

    @abstractmethod
    def dispatch(self, sim, call: Call) -> int | None:
        """Pick an ambulance id for `call`, or None to let the call wait in the queue."""

    @abstractmethod
    def reposition(self, sim, amb_id: int) -> int:
        """Pick the base node that ambulance `amb_id` should drive to now."""

    def reset(self) -> None:
        """Clear any per-episode internal state. Called before each run."""
