# Ambulance Dispatch and Repositioning

AI course project (Assignment 2) — Anuj Trivedi, B23CH1006, IIT Jodhpur.

Emergency calls arrive stochastically across a city road network. A fleet of ambulances
must serve them, and an agent makes two coupled decisions:

- **Dispatch** — when a call comes in, which available unit to send (or whether to let it wait).
- **Repositioning** — where each idle unit should wait, re-decided as the fleet state changes.

The two conflict. Sending the nearest unit is optimal for the call in front of you, but it
can strip cover from a dense zone and cost the next call far more than it saved this one.
The project builds a simulation testbed for that trade-off and compares classical
heuristics, online search, and learned policies on it.

See [ROADMAP.md](ROADMAP.md) for the formulation, the phase plan, and the decision log.

## Model

Decisions occur only at event epochs, so the problem is modelled as a **semi-Markov
decision process** over a discrete-event simulation:

| | |
|---|---|
| State | time, per-unit (status, position, ETA), queue of unserved calls |
| Action | which unit to dispatch; which base an idle unit drives to |
| Stochasticity | time-varying Poisson arrivals, spatial demand hotspots, noisy travel times, random on-scene durations |
| Objective | maximise calls reached within 8 minutes; report mean/p90 response time and worst-zone coverage |

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_baseline.py --days 7 --reps 10
python scripts/run_phase2.py --days 7 --reps 30
python -m pytest tests -q
```

## Current results

8x8 grid city, 4 stations, 6 ambulances, 4 calls/hour, 7 simulated days x 30 paired
replications (~648 calls per replication):

| policy | coverage (≤8 min) | mean response | p90 response | worst-zone coverage |
|---|---|---|---|---|
| B0 random (floor) | 0.357 | 10.09 min | 16.21 min | 0.282 |
| B1 nearest-idle, hand-placed stations | 0.765 | 5.82 min | 11.72 min | 0.727 |
| B2a nearest-idle, p-median stations | 0.858 | **4.67 min** | 9.73 min | 0.792 |
| B2b nearest-idle, MEXCLP stations | 0.860 | 5.38 min | 9.14 min | 0.798 |
| B3 DMEXCLP on MEXCLP stations | **0.872** | 5.26 min | **8.81 min** | **0.822** |
| B3h DMEXCLP on hand-placed stations | 0.836 | 5.07 min | 10.31 min | 0.711 |

B3 beats B1 on all four metrics, every one significant at 5% on both a paired t-test and
Wilcoxon signed-rank (n = 30). The interesting part is the decomposition:

- **Station placement does most of the work.** B1→B2b is +0.096 coverage; adding dynamic
  repositioning on top (B2b→B3) adds only +0.012. Repositioning on *badly placed* stations
  (B1→B3h) is worth +0.071 on its own — so the two are largely substitutes, and a large
  reported DMEXCLP gain is implicitly a report of badly-placed stations.
- **DMEXCLP trades equity for the average.** B3h lifts city-wide coverage by +0.071 while
  *lowering* worst-zone coverage by 0.016 (p ≈ 0.08 — a hint, not yet a result). Its
  objective has no term for the worst-off zone. Phase 5 picks this up.
- **p-median and MEXCLP optimise different things**, exactly as theory says: p-median wins
  the mean (4.67 min), MEXCLP wins the tail and coverage (p90 9.14 min).

Reproduce with `python scripts/run_phase2.py --reps 30 --days 7`.

## Layout

```
src/ambulance/
  city.py            road graph, travel times, demand model, call generation
  sim.py             discrete-event simulator and ambulance state machine
  location.py        static station placement: p-median and MEXCLP
  metrics.py         response-time, coverage and equity metrics
  experiment.py      paired, seeded replications
  stats.py           paired t + Wilcoxon significance testing
  policies/          one Policy interface, every method behind it
scripts/run_baseline.py   Phase 1 reference policies
scripts/run_phase2.py     station optimisation + dynamic repositioning
tests/
```
