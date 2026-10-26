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

- [x] B1 — **Nearest-idle dispatch, static home base** (what most real services do)
- [x] B2 — **Static optimal base locations**: p-median and MEXCLP (maximum expected
      covering location) solved by greedy + swap local search; dispatch still nearest-idle
- [x] B3 — **DMEXCLP dynamic repositioning**: when a unit frees, send it to the base that
      maximises marginal expected covered demand given the current busy set
- [x] Significance testing across 30 seeded replications (paired t + Wilcoxon)
- [x] Ablation: DMEXCLP on hand-placed stations, separating repositioning from placement

**Deliverable:** `python scripts/run_phase2.py --reps 30`. B3 beats B1 by +0.107 coverage,
all four metrics significant on both tests. Two findings to carry into the report:

1. **Placement does most of the work.** Optimising stations (B1→B2b) buys +0.096 coverage;
   adding dynamic repositioning on top (B2b→B3) buys only +0.012 more. On *hand-placed*
   stations, repositioning alone buys +0.071 — so the two are largely substitutes, and
   DMEXCLP's headline benefit depends heavily on how bad the static layout was.
2. **DMEXCLP trades equity for the mean.** On hand-placed stations it improves city-wide
   coverage by +0.071 while *reducing* worst-zone coverage by 0.016 (p ≈ 0.08, so a hint
   rather than a result). Its objective is aggregate expected coverage, with no term for
   the worst-off zone. Phase 5's equity analysis picks this thread up.

Also as theory predicts: p-median wins on mean response (4.67 vs 5.38 min) while MEXCLP
wins on the tail and on coverage (p90 9.14 vs 9.73). They optimise different things, so the
report shows both rather than picking one.

## Phase 3 — Online planning / search

- [x] One-step lookahead with a **Monte Carlo rollout** of a base policy (B3) as the
      value estimate — the classic rollout algorithm
- [x] **Open-loop MCTS** over sequences of repositioning actions, UCT with normalised values
- [x] Anytime behaviour study: performance vs. simulation budget per decision
- [x] Confirmation run at 20 reps x 3 days, to tell a real null from a weak test

**Deliverable:** `results/figures/phase3_anytime.png`. The honest result is negative, and
it is the most useful finding in the project so far.

1. **Neither method beat its own base policy, at any budget.** Coverage stays within noise
   of DMEXCLP from 1 to 64 simulated futures per decision, while costing 84x to 3460x more
   compute. The curve is flat: 32x more search buys nothing measurable.
2. **It is a real null, not low power.** The confirmation run puts the coverage difference
   at -0.001 with 95% CI [-0.008, +0.006] — tight enough to exclude an effect the size of
   DMEXCLP's own Phase 2 gain (+0.012).
3. **The reason is estimator variance, measured directly.** The coverage difference between
   top candidate stations is ~0.005; the paired standard deviation across sampled futures is
   ~0.035 even with common random numbers. That needs ~200 futures per candidate per
   decision, against 1–32 affordable. The search overrides an informative base-policy
   ranking with an estimate that is mostly noise.
4. **MCTS ≈ rollout.** Allocating the budget unevenly by UCT changes nothing, which fits (3):
   the limit is per-sample noise, not how samples are divided. Open-loop node blurring should
   cost MCTS something but is invisible at this noise floor — these two explanations cannot be
   separated from this data, so neither is claimed.
5. **Reward design beat search depth.** Rollout is significantly *worse* on mean response
   (+0.077 min, p = 0.004, d = 0.73), and all twelve search configurations lean that way while
   coverage scatters. The binary 8-minute value function cannot tell a 3-minute response from
   a 7-minute one, so the search trades fast responses away for free.

This is what motivates Phase 4. The fix for a noisy Monte Carlo value estimate is not more
samples at decision time — it is to amortise the estimate offline into a learned value
function, which is exactly what ADP and RL do.

## Phase 4 — Learning

- [x] Feature-based **approximate value function** (ADP) over post-decision states, fit by
      least-squares temporal differences; plus LSPI on top of it
- [x] **Direct policy search** (cross-entropy) over the *same* linear class, as a control
      that isolates the objective from the representation
- [x] **Deep RL**: neural fitted value iteration over post-decision states, richer features
      (no Gymnasium needed — the simulator is already the environment)
- [x] Ablation on reward shaping (continuous response time vs. the target indicator)
- [ ] Generalisation test: train on one demand profile, evaluate on a shifted one

**Deliverable:** `results/figures/phase4_reward.png`. The answer to "does learning beat the
heuristic" is **no — the best learned policy ties it**, and the interesting part is why.

1. **The hypothesis class contains the heuristic.** Greedy on the expected-covered-demand
   feature is algebraically DMEXCLP's rule, and scores identically to nine decimal places
   (pinned by a test). A hand-set "coverage − distance" weighting beats it. So any learner
   that loses here is losing on the training signal, not the features.
2. **Every value-fitting method loses anyway.** LSTD −0.063 coverage, NFQ −0.148, LSPI a
   wash. All three fit value *prediction*, and the directions an action can move the state
   contribute almost nothing to prediction error, which exogenous call variation dominates.
   Nothing in the objective rewards ranking actions correctly.
3. **Capacity made it worse.** The network beats the linear model on prediction loss and
   loses badly on policy quality — it fits the exogenous noise faster.
4. **Direct policy search ties the heuristic** using the same three weights, confirming (1)
   and (2): change the objective from prediction to policy quality and the gap closes.
   It does not *beat* the heuristic, and its training score (0.907) overstates its test
   score (0.871) — selection bias from picking the best of 20 noisy candidates.
5. **The reward dominated everything.** Binary to continuous is worth +0.052 coverage
   (linear) and +0.148 (network), larger than any architectural choice tested. The binary
   reward also collapses worst-zone coverage from 0.822 to 0.384: maximising a count of
   indicators is maximised by serving dense zones and abandoning sparse ones.

Phases 2, 3 and 4 converge on one lesson from three directions — the objective you write
down matters more than the machinery you point at it.

## Phase 5 — Analysis & extensions (pick 2)

- [x] **Priority classes** and *selective dispatch* — is it ever right to hold a unit back?
- [x] **Equity vs. efficiency** Pareto frontier (city-wide mean vs. worst-zone performance)

**Deliverable:** `results/figures/phase5_tradeoffs.png`. One trade-off is real and badly
priced; the other turns out not to exist.

1. **Selective dispatch works and is a bad deal.** Reserving two units is the only setting
   with a significant urgent-coverage gain (+0.009, p = 0.004), bought for −0.116 routine
   coverage (p = 6.5e-21) — about 13 routine points per urgent point. Reserving four is
   worse than reserving none for *both* classes, because the queue backs up far enough to
   delay the urgent calls the reserve was protecting.
2. **No equity/efficiency frontier exists at this configuration.** Every λ from 0 to 1 lands
   within ~2 standard errors of plain DMEXCLP on both axes; the best moves worst-zone
   coverage by +0.001 (p = 0.92) and λ = 1 is significantly worse city-wide.
3. **And not because the knob is inert — that was measured, not assumed.** λ = 1 changes
   41% of 1,248 repositioning decisions, λ = 0.7 changes 26%. Behaviour moves a lot; results
   do not move at all.
4. **Which is Phase 3's finding from the other side.** The candidate stations differ by
   ~0.005 coverage against ~0.035 of noise, so search cannot resolve them (Phase 3) *and*
   equity cannot be traded against efficiency (Phase 5). Same cause, different symptom.
   This caveat has since been tested — see the regime sweep below — and it did not hold up
   the way it was written. **Correction:** this roadmap previously predicted that "a finer
   station set would give the lever something to pull". The opposite is true. Going from 4
   to 8 stations *shrinks* the gap between the best and second-best candidate from 10.0% to
   6.6% of the best gain, and makes rollout significantly worse. More options mean more
   near-equivalent options.

## Regime sweep — is any of this an artefact of one configuration?

Phases 3, 4 and 5 all concluded that sophisticated methods do not pay here, and all three
rested on the unverified claim above. Two one-dimensional sweeps through the centre point
turn it into a measurement, with two quantities kept deliberately apart:

- **Headroom** — DMEXCLP minus random parking, dispatch held fixed: what the repositioning
  choice is worth at all.
- **Best vs second-best gain** — what a search method actually has to resolve, since it
  never chooses between the best option and a random one.

| regime | queued | headroom | best vs 2nd | rollout Δ | equity Δ |
|---|---|---|---|---|---|
| 4/h, 4 stations | 0.062 | +0.091 | 10.0% | −0.008 | −0.007 |
| 5/h, 4 stations | 0.135 | +0.061 | 10.2% | +0.001 | −0.005 |
| 6/h, 4 stations | 0.283 | +0.039 | 10.5% | −0.005 | −0.012 |
| 4/h, 6 stations | 0.064 | +0.071 | 7.0% | −0.012 | −0.003 |
| 4/h, 8 stations | 0.067 | +0.078 | **6.6%** | **−0.016*** | −0.007 |

- [x] **The negatives are not a configuration artefact.** Rollout fails to beat DMEXCLP in
      every regime tested, across a 1.5x range of load and a 2x range of station count. The
      equity weight has no significant effect anywhere.
- [x] **Headroom is large but falls with load** (+0.091 at 65% utilisation to +0.039 near
      saturation). Repositioning matters *less* when the system is busy, because units are
      rarely idle and where they would have waited stops mattering.
- [x] **Repositioning matters; choosing between the top two does not.** Headroom of +0.091
      says the choice is worth nine coverage points, while the top two candidates differ by
      10% of one gain. Both are true, and only the second governs whether search can pay —
      which is why reporting headroom alone would tell the opposite story.
- [x] **A finer action space makes search worse, not better.** At 8 stations rollout is
      significantly *harmful* (−0.016, p = 0.008): more near-equivalent candidates mean more
      chances for a noisy estimate to pick a worse one. Exactly what Phase 3's variance
      argument predicts, now observed rather than inferred.

**Power caveat:** this sweep is 8 replications x 3 days, far less than the 30 x 7 behind the
phase results. Its null results have confidence intervals around ±0.02, so they exclude
large effects, not small ones. The one significant finding is a harm, not a benefit.
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
| 2026-10-25 | Greedy + swap local search for p-median/MEXCLP, not an ILP | Lands on or near the optimum at these graph sizes with no solver dependency |
| 2026-10-25 | MEXCLP allocates units with multiplicity; stations stay distinct | Doubling up on a hotspot is the model's whole point, so the simulator takes an explicit home-base allocation |
| 2026-10-25 | Report paired t *and* Wilcoxon, require both to agree | Coverage differences are near-symmetric but response-time differences are not; a result only one test supports is not stated as a result |
| 2026-10-25 | Fixed q = 0.35 in the coverage model, not estimated from the sim | Keeps B3 a pure heuristic with nothing tuned on the test scenarios; adapting q is a Phase 4 comparison point |
| 2026-10-26 | Forks resample call arrivals instead of replaying the real ones | A fork carrying the real pending calls would let the policy see the future it is graded on; asserted by test rather than left to care |
| 2026-10-26 | Every candidate scored on the same sampled futures (common random numbers) | Cuts the paired standard deviation from 0.17 to 0.07; without it the comparison is hopeless rather than merely hard |
| 2026-10-26 | Candidates shortlisted by the base policy's own ranking | Stops budget going to options already known to be poor, and makes an uninformative search degrade to the base policy instead of below it |
| 2026-10-26 | MCTS is open-loop (nodes keyed by action sequence, not state) | Each iteration samples its own future, so a state-keyed node would never see a second visit |
| 2026-10-26 | Plot against simulated futures, not the budget parameter | Rollout spends candidates x sims per decision and MCTS spends sims; the parameter would flatter MCTS by the shortlist size |
| 2026-10-26 | Report the negative Phase 3 result rather than tuning for a win | The null is tight, mechanistically explained, and it is what motivates Phase 4 |
| 2026-10-26 | Value function over post-decision states, not Q over (state, action) | The post-decision state is deterministic given (s, a), so ranking actions needs one cheap vector each and no per-station output head that would not transfer between cities |
| 2026-10-26 | Feature set cut from 12 to 7 after checking the design matrix | Three features were exactly redundant and four more were collinear at 0.97-1.00; ridge hid it, but the greedy policy moves along exactly the unidentified directions |
| 2026-10-26 | CEM over the same linear class kept as a control, not as a method | It is the only way to separate "the features are wrong" from "fitting value does not rank actions"; without it the ADP result is uninterpretable |
| 2026-10-26 | Deep RL written directly against the simulator, no Gymnasium | The simulator already is the environment; a wrapper would add a dependency and an adapter layer for nothing |
| 2026-10-26 | Deep RL swept over rewards and buffers before being reported | One hyperparameter setting said "deep RL fails"; the sweep showed the reward was doing the work and the architecture was not |
| 2026-10-26 | Equity weight measured for how often it changes the decision | A flat set of outcomes has two opposite explanations — an inert knob, or a knob whose choices do not matter. Measuring it showed the second, and the first was the wrong guess |
| 2026-10-26 | Phase 5's equity panel drawn with a noise band, not a Pareto front | Marking non-dominated points would manufacture a trade-off curve out of sampling noise; every point is inside ±2 s.e. of λ=0 |
| 2026-10-26 | Selective dispatch withholds only while a unit is busy | Otherwise no BECOME_FREE event ever fires and a held call waits out the run: a deadlock, not a trade-off. Covered by a test |
| 2026-10-25 | Greedy + swap local search for p-median/MEXCLP, not an ILP | Lands on or near the optimum at these graph sizes with no solver dependency |
| 2026-10-25 | MEXCLP allocates units with multiplicity, stations stay distinct | Doubling up on a hotspot is the model's whole point; the simulator takes an explicit home-base allocation so it can be reproduced |
| 2026-10-25 | Report paired t *and* Wilcoxon, require both | Coverage differences are near-symmetric but response-time differences are not; a result only one test supports is not stated as a result |
| 2026-10-25 | Fixed q = 0.35 in the coverage model, not an estimate from the sim | Keeps B3 a pure heuristic with no tuning; adapting q is a Phase 4 comparison point |
