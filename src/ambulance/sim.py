"""Discrete-event simulator for ambulance dispatch and repositioning."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .city import Call, City, generate_calls

HOSPITAL_HANDOVER_MINUTES = 15.0

# Hypothetical calls sampled inside a fork get ids from here up, so they can never
# collide with the real call stream's ids.
_FORK_ID_BASE = 10_000_000


class Status(Enum):
    IDLE = auto()        # parked at a base
    RELOCATING = auto()  # driving to a base, still dispatchable
    TO_SCENE = auto()
    AT_SCENE = auto()
    TO_HOSPITAL = auto()
    AT_HOSPITAL = auto()


DISPATCHABLE = {Status.IDLE, Status.RELOCATING}


class EventType(Enum):
    CALL_ARRIVAL = auto()
    ARRIVE_SCENE = auto()
    DEPART_SCENE = auto()
    ARRIVE_HOSPITAL = auto()
    BECOME_FREE = auto()
    ARRIVE_BASE = auto()


@dataclass(order=True)
class Event:
    time: float
    seq: int
    type: EventType = field(compare=False)
    amb_id: int | None = field(default=None, compare=False)
    call: Call | None = field(default=None, compare=False)
    trip: int = field(default=-1, compare=False)


@dataclass
class Ambulance:
    """One unit. While moving, its position is interpolated along its route."""

    id: int
    home_base: int
    node: int                       # last node passed
    status: Status = Status.IDLE
    route: list[int] = field(default_factory=list)          # nodes still to traverse
    route_times: list[float] = field(default_factory=list)  # absolute arrival time each
    call: Call | None = None
    trip: int = 0  # bumped on every new activity; stale events are discarded

    def position(self, t: float) -> tuple[int, float]:
        """Return (next node this unit will reach, minutes until it reaches it).

        A stationary unit is already at `node`, with zero delay. This is what lets a
        relocating unit be dispatched from mid-route rather than from where it started.
        """
        for node, arrive in zip(self.route, self.route_times):
            if arrive >= t:
                return node, arrive - t
        return self.node, 0.0

    @property
    def destination(self) -> int:
        """Where this unit will end up: its target if moving, else where it stands.

        Coverage-aware policies reason about a relocating unit at its destination base,
        not at its current position, since that is where it will be waiting.
        """
        return self.route[-1] if self.route else self.node


@dataclass
class CallRecord:
    call: Call
    assigned_at: float | None = None
    amb_id: int | None = None
    reached_at: float | None = None

    @property
    def response_time(self) -> float | None:
        if self.reached_at is None:
            return None
        return self.reached_at - self.call.arrival_time

    @property
    def wait_to_assign(self) -> float | None:
        if self.assigned_at is None:
            return None
        return self.assigned_at - self.call.arrival_time


def _copy_record(rec: CallRecord) -> CallRecord:
    """Copy a call record. `Call` itself is frozen, so it can be shared."""
    return CallRecord(
        call=rec.call,
        assigned_at=rec.assigned_at,
        amb_id=rec.amb_id,
        reached_at=rec.reached_at,
    )


class Simulator:
    """Event-driven simulation of one scenario under one policy.

    The policy is queried at exactly the decision epochs of the underlying SMDP: a call
    arrival (dispatch) and a unit becoming available (reposition).
    """

    def __init__(
        self,
        city: City,
        calls: list[Call],
        n_ambulances: int = 4,
        seed: int = 0,
        travel_noise: float = 0.15,
        home_bases: list[int] | None = None,
    ) -> None:
        self.city = city
        self.calls = sorted(calls, key=lambda c: c.arrival_time)
        self.rng = np.random.default_rng(seed)
        self.travel_noise = travel_noise
        self.time = 0.0
        self._seq = 0
        self._events: list[Event] = []
        self.queue: list[CallRecord] = []
        self.records: dict[int, CallRecord] = {}

        # Round-robin over the stations unless an explicit allocation is given, which is
        # how MEXCLP homes several units at one station.
        homes = list(home_bases) if home_bases else [
            city.bases[i % len(city.bases)] for i in range(n_ambulances)
        ]
        if len(homes) != n_ambulances:
            raise ValueError(
                f"home_bases has {len(homes)} entries for {n_ambulances} ambulances"
            )
        self.ambulances = [
            Ambulance(id=i, home_base=homes[i], node=homes[i])
            for i in range(n_ambulances)
        ]

    # ---------------------------------------------------------------- scheduling

    def _push(self, time: float, type_: EventType, **kw) -> None:
        self._seq += 1
        heapq.heappush(self._events, Event(time, self._seq, type_, **kw))

    def _retire(self, amb: Ambulance) -> int:
        """Start a new activity for `amb`, invalidating events from the previous one.

        A relocating unit can be dispatched mid-route, which leaves its ARRIVE_BASE event
        sitting in the heap; without this, that stale event would later mark the unit idle
        while it is actually on a call.
        """
        amb.trip += 1
        return amb.trip

    def _drive(self, amb: Ambulance, dest: int) -> float:
        """Send `amb` toward `dest`; fill in its route and return the arrival time."""
        start, delay = amb.position(self.time)
        path = self.city.path(start, dest)
        noise = max(0.6, 1.0 + self.travel_noise * self.rng.normal())
        t = self.time + delay
        amb.route, amb.route_times = [], []
        for u, v in zip(path, path[1:]):
            t += self.city.graph.edges[u, v]["time"] * noise
            amb.route.append(v)
            amb.route_times.append(t)
        if not amb.route:  # already there
            amb.node = dest
            return self.time + delay
        return amb.route_times[-1]

    def travel_time_to(self, amb: Ambulance, node: int) -> float:
        """Expected minutes for `amb` to reach `node` from where it is right now."""
        here, delay = amb.position(self.time)
        return delay + self.city.travel_time(here, node)

    # ------------------------------------------------------------------ actions

    def assign(self, amb: Ambulance, record: CallRecord) -> None:
        amb.call = record.call
        amb.status = Status.TO_SCENE
        trip = self._retire(amb)
        record.amb_id = amb.id
        record.assigned_at = self.time
        eta = self._drive(amb, record.call.node)
        self._push(eta, EventType.ARRIVE_SCENE, amb_id=amb.id, call=record.call, trip=trip)

    def send_to_base(self, amb: Ambulance, base: int) -> None:
        here, delay = amb.position(self.time)
        trip = self._retire(amb)
        if here == base and delay == 0.0:
            amb.status, amb.node = Status.IDLE, base
            amb.route, amb.route_times = [], []
            return
        amb.status = Status.RELOCATING
        arrival = self._drive(amb, base)
        self._push(arrival, EventType.ARRIVE_BASE, amb_id=amb.id, trip=trip)

    # -------------------------------------------------------------------- state

    def available(self) -> list[Ambulance]:
        return [a for a in self.ambulances if a.status in DISPATCHABLE]

    def busy_fraction(self) -> float:
        return 1.0 - len(self.available()) / len(self.ambulances)

    # --------------------------------------------------------------- event loop

    def run(self, policy, horizon: float | None = None) -> list[CallRecord]:
        for call in self.calls:
            self.records[call.id] = CallRecord(call)
            self._push(call.arrival_time, EventType.CALL_ARRIVAL, call=call)
        self.resume(policy, until=horizon)
        return [self.records[c.id] for c in self.calls]

    def resume(self, policy, until: float | None = None) -> None:
        """Process events from the current state up to `until`.

        Separated from `run` so a forked simulator, whose events are already in the heap,
        can be advanced without re-seeding the call stream.
        """
        while self._events:
            if until is not None and self._events[0].time > until:
                break
            ev = heapq.heappop(self._events)
            self.time = ev.time
            self._handle(ev, policy)

    # ------------------------------------------------------------------ lookahead

    def fork(self, rng: np.random.Generator, horizon: float) -> "Simulator":
        """A copy of the current state with a freshly *sampled* future.

        Lookahead policies need to simulate forward from the present decision epoch. The
        copy therefore keeps everything already committed -- unit positions, missions in
        flight, calls still queued -- but drops every pending call arrival and samples a
        new stream over the next `horizon` minutes from the demand model.

        Dropping the real arrivals is the whole point. Keeping them would let the policy
        see the actual future and score far better than it could in deployment, which is
        the standard way a lookahead experiment becomes meaningless.
        """
        clone = object.__new__(Simulator)
        clone.city = self.city  # shared: read-only during a run
        clone.calls = []
        clone.rng = rng
        clone.travel_noise = self.travel_noise
        clone.time = self.time
        clone._seq = self._seq
        clone.ambulances = [
            Ambulance(
                id=a.id,
                home_base=a.home_base,
                node=a.node,
                status=a.status,
                route=list(a.route),
                route_times=list(a.route_times),
                call=a.call,
                trip=a.trip,
            )
            for a in self.ambulances
        ]

        # Fresh records for anything still in play, so the real run is never mutated.
        clone.records = {}
        clone._events = []
        for ev in self._events:
            if ev.type is EventType.CALL_ARRIVAL:
                continue  # the real future stays hidden
            if ev.call is not None and ev.call.id not in clone.records:
                clone.records[ev.call.id] = _copy_record(self.records[ev.call.id])
            heapq.heappush(clone._events, ev)
        clone.queue = []
        for rec in self.queue:
            copied = clone.records.setdefault(rec.call.id, _copy_record(rec))
            clone.queue.append(copied)

        clone.horizon_start = self.time
        clone.sampled_ids = set()
        for call in generate_calls(
            self.city, horizon, rng, start_time=self.time, first_id=_FORK_ID_BASE
        ):
            clone.records[call.id] = CallRecord(call)
            clone.sampled_ids.add(call.id)
            clone._push(call.arrival_time, EventType.CALL_ARRIVAL, call=call)
        return clone

    def _handle(self, ev: Event, policy) -> None:
        if ev.amb_id is not None and ev.trip != self.ambulances[ev.amb_id].trip:
            return  # stale: this unit has been given a new assignment since

        if ev.type is EventType.CALL_ARRIVAL:
            rec = self.records[ev.call.id]
            choice = policy.dispatch(self, rec.call)
            if choice is None:
                self.queue.append(rec)
            else:
                self.assign(self.ambulances[choice], rec)

        elif ev.type is EventType.ARRIVE_SCENE:
            amb = self.ambulances[ev.amb_id]
            amb.node, amb.route, amb.route_times = ev.call.node, [], []
            amb.status = Status.AT_SCENE
            self.records[ev.call.id].reached_at = self.time
            self._push(
                self.time + ev.call.on_scene_time,
                EventType.DEPART_SCENE,
                amb_id=amb.id,
                call=ev.call,
                trip=amb.trip,
            )

        elif ev.type is EventType.DEPART_SCENE:
            amb = self.ambulances[ev.amb_id]
            if ev.call.transport_required:
                amb.status = Status.TO_HOSPITAL
                hosp = min(
                    self.city.hospitals,
                    key=lambda h: self.city.travel_time(amb.node, h),
                )
                arrival = self._drive(amb, hosp)
                self._push(
                    arrival,
                    EventType.ARRIVE_HOSPITAL,
                    amb_id=amb.id,
                    call=ev.call,
                    trip=amb.trip,
                )
            else:
                self._push(self.time, EventType.BECOME_FREE, amb_id=amb.id, trip=amb.trip)

        elif ev.type is EventType.ARRIVE_HOSPITAL:
            amb = self.ambulances[ev.amb_id]
            amb.node = amb.route[-1] if amb.route else amb.node
            amb.route, amb.route_times = [], []
            amb.status = Status.AT_HOSPITAL
            self._push(
                self.time + HOSPITAL_HANDOVER_MINUTES,
                EventType.BECOME_FREE,
                amb_id=amb.id,
                trip=amb.trip,
            )

        elif ev.type in (EventType.BECOME_FREE, EventType.ARRIVE_BASE):
            amb = self.ambulances[ev.amb_id]
            if ev.type is EventType.ARRIVE_BASE:
                amb.node = amb.route[-1] if amb.route else amb.node
            amb.route, amb.route_times = [], []
            amb.call = None
            amb.status = Status.IDLE
            if not self._serve_queue(policy, amb):
                base = policy.reposition(self, amb.id)
                self.send_to_base(amb, base)

    def _serve_queue(self, policy, amb: Ambulance) -> bool:
        """Give a newly free unit to the most urgent waiting call, if any."""
        if not self.queue:
            return False
        self.queue.sort(key=lambda r: (r.call.priority, r.call.arrival_time))
        rec = self.queue.pop(0)
        self.assign(amb, rec)
        return True
