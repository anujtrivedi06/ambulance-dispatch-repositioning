# Roadmap — Ambulance Dispatch & Repositioning

Project log and plan. Each phase is sized to land as 3–8 commits so the history shows
continuous development rather than one dump.

## 0. Problem formulation

Emergency calls arrive stochastically over a city road network. A fleet of `K` ambulances
must serve them. The agent makes two coupled decisions:

1. **Dispatch** — when a call arrives, which available ambulance (if any) to send.
2. **Repositioning** — where each idle ambulance should wait, re-decided whenever the
   fleet state changes (a unit goes busy, or a unit becomes free at a hospital).

These are coupled: sending the nearest unit is myopically optimal for *this* call but may
strip coverage from a dense zone, hurting the *next* call. That tension is the project.

### Formal model — semi-Markov decision process

Decisions are made only at *event epochs* (call arrival, or unit becoming free), so the
natural model is an SMDP solved over a discrete-event simulation.

- **State** `s = (t, {(status_k, loc_k, eta_k)}_{k=1..K}, queue of waiting calls)`
  where `status ∈ {idle_at_base, relocating, to_scene, at_scene, to_hospital, at_hospital}`.
- **Action**
  - at a call arrival: `a ∈ {dispatch unit k} ∪ {queue the call}`
  - at a unit becoming free: `a ∈ {bases}` (which base to drive to)
- **Transition** — exogenous call arrivals (time-varying Poisson), stochastic travel and
  on-scene/hospital service times.
- **Reward** — negative response time, or the standard EMS target
  `r = +1 if response_time ≤ T_target else 0` (T_target = 8 min urban convention).
- **Objective** — maximise expected fraction of calls reached within target, with mean and
  90th-percentile response time and inter-zone equity as secondary metrics.

### Why this is an AI project, not an OR exercise
The state space is continuous and combinatorial (≈ `(|V| × 6)^K`), so exact DP is hopeless.
It needs the approximation toolkit: heuristic policies, online search/rollout, and learned
value functions — and a fair comparison between them.

---

## Phase 1 — Simulation testbed  *(the foundation; get this right)*

- [x] City graph: synthetic grid + travel-time edge weights, all-pairs shortest paths
- [x] Demand model: node-level demand weights + time-of-day arrival-rate profile
- [x] Discrete-event simulator: event queue, ambulance state machine, call lifecycle
- [x] Metrics: response-time distribution, % within target, utilisation, per-zone equity
- [ ] Reproducibility: seeded RNG streams, scenario config files (YAML/JSON)
- [ ] Replay log + matplotlib visualisation of one simulated day
- [x] Unit tests for graph, call stream, and simulator invariants

**Deliverable:** `python scripts/run_baseline.py` prints a metrics table for one scenario.

## Phase 2 — Classical baselines

- [ ] B1 — **Nearest-idle dispatch, static home base** (what most real services do)
- [ ] B2 — **Static optimal base locations**: p-median and MEXCLP (maximum expected
      covering location) solved by greedy + local search; dispatch still nearest-idle
- [ ] B3 — **DMEXCLP dynamic repositioning**: when a unit frees, send it to the base that
      maximises marginal expected covered demand given the current busy set
- [ ] Significance testing across ≥30 seeded replications (paired, same call stream)

**Deliverable:** baselines table; B3 should already beat B1 clearly. This is the bar to beat.

## Phase 3 — Online planning / search

- [ ] One-step lookahead with a **Monte Carlo rollout** of a base policy (B3) as the
      value estimate — the classic rollout algorithm
- [ ] **MCTS** over dispatch+reposition actions with a progressive-widening action set
- [ ] Anytime behaviour study: performance vs. simulation budget per decision

**Deliverable:** plot of solution quality vs. compute budget. Shows the search–quality trade-off.

## Phase 4 — Learning

- [ ] Feature-based **approximate value function** (ADP): features = per-zone coverage
      deficit, busy fraction, time of day; fit by least-squares temporal differences
- [ ] **Deep RL**: Gymnasium wrapper over the simulator, masked-action DQN or PPO on a
      compact tensor state encoding
- [ ] Learning curves, ablation on reward shaping (response time vs. target indicator)
- [ ] Generalisation test: train on one demand profile, evaluate on a shifted one

**Deliverable:** does learning beat the DMEXCLP heuristic? An honest negative result here is
a legitimate and defensible finding — report it either way.

## Phase 5 — Analysis & extensions (pick 2)

- [ ] **Priority classes** and *selective dispatch* — is it ever right to hold a unit back?
- [ ] **Equity vs. efficiency** Pareto frontier (city-wide mean vs. worst-zone performance)
- [ ] **Fleet sizing**: marginal value of the Nth ambulance
- [ ] **Real city graph** via OSMnx (Jodhpur) with time-of-day traffic multipliers
- [ ] **Robustness**: performance under a mass-casualty demand spike

## Phase 6 — Write-up

- [ ] Report: formulation, methods, experiments, results, limitations
- [ ] Reproduce-everything script + final figures
- [ ] README with results summary

---

## Design decisions (log)

| Date | Decision | Reason |
|---|---|---|
| 2026-10-25 | Discrete-event SMDP sim, not fixed-timestep grid | Decisions are naturally event-driven; a timestep sim wastes compute and distorts travel times |
| 2026-10-25 | Synthetic grid graph first, OSMnx later | Removes a data dependency from the critical path; the policy code is graph-agnostic |
| 2026-10-25 | Policy = two methods (`dispatch`, `reposition`) behind one ABC | Lets every method from heuristic to RL slot into the same evaluation harness |
