# Independent adversarial review

Review date: 2026-10-25. This review is performed independently of implementation and benchmark design. Findings below distinguish design risks from demonstrated failures; no behavioral success is presumed.

## Claims the implementation must survive

1. Durable storage is necessary but does not establish functional affect. Interventions on saved state must change nonlinguistic control decisions while preserving observations and permissions.
2. Affective names do not establish a distinct mechanism. An otherwise identical no-label condition should have identical behavior. Improvement over a weaker nonadaptive condition establishes adaptation, not a special benefit from affect.
3. Predicting resource consequences is close to ordinary model predictive control and homeostatic reinforcement learning. A research novelty claim requires a precise difference from those mechanisms and checked prior art.
4. The procedural benchmark can establish search-controller behavior in its specified simulator; it cannot establish real-model creativity, human usefulness, published novelty, biological fidelity, or subjective experience.
5. Development requires the same later observation to acquire different appraisal or coping because of earlier evidence, with generic trained weights fixed. Mere accumulation of mood or event counts is insufficient.
6. Equal numbers of external proposals are not equal compute if one condition uses extra model calls or hidden oracle evaluations. Internal prediction, appraisal, reflection, and counterfactual computation must be counted.

## Invariants and adversarial histories

| Invariant | Adversarial case | Required result |
|---|---|---|
| Event idempotence | Repeated ID and identical payload | No new affect, memory, learning, or event-time advance |
| Identity integrity | Same ID with altered evidence | Reject rather than silently accepting conflict |
| Evidence separation | Recall/quotation/prediction of failure | May reactivate affect; must not count as a new observed failure |
| Outcome integrity | Repeated recording of an outcome | No second adaptation or reward |
| Restart consistency | Reopen the same store | Exact numerical and adaptive state retained |
| Agent isolation | Same event ID in another agent | Independent legal identity; no state leakage |
| Fork isolation | Change a fork | Parent and original snapshot remain unchanged |
| Replay closure | Replay after an intervention, restore, or reappraisal | Consume recorded operations and provider outputs without live calls |
| Numerical integrity | NaN/infinite confidence, time, gains | Reject before storage; no partially committed operation |
| Temporal integrity | Negative elapsed time or reordered events | Declared handling, no accidental recovery or exponential explosion |
| Social validity | Praise without success | Affiliation may change; objective quality cannot become success |
| Evidence standards | Misleading criticism without supporting outcome | Preserve alternatives and uncertainty, no fact promotion |
| Memory feedback | Persistent stress plus selective retrieval | Stable recovery under cessation; bounds alone are insufficient |
| Development | Same new observation before/after validated history | Predictable change in learned expectation or coping |

## Benchmark adversarial checks

The primary comparison must be learned/hybrid regulation versus an ordinary adaptive controller sharing information, action space, parameter capacity, pretrained data, episode histories, and external search budget. A same-mechanism no-label clone identifies effects of names only; it is not a substitute for the structurally different ordinary adaptive baseline.

Frozen, shuffled, and disconnected-memory interventions identify causal dependencies only when all unaffected inputs are held fixed. Shuffling should use a declared permutation independent of outcomes and should not accidentally add information from future observations.

Test episodes must begin from declared complete snapshots. Held-out evaluation may adapt only within an episode if the protocol allows it. Learned shared weights must not update between test episodes. Seeds and task families must be selected before result inspection; reports should include paired uncertainty estimates and all conditions, including negative outcomes.

Independent executable checks should evaluate constraint compliance, actual utility, and mechanism differences against an explicit corpus. Diversity over a finite grammar is benchmark-relative combinatorial diversity. A score generated from the same grammar used to produce candidate designs is useful simulator feedback, not independent validation of original engineering ideas.

## Execution findings

Executed independently with `python -m unittest tests.test_adversarial -v`: **31 tests passed** on 2026-10-25. These fixtures cover finite-number boundaries, evidence mutation after dataclass construction, idempotent observations, conflicting identities, duplicate outcomes, quotation outcome rejection, process reopening, per-agent isolation, snapshot integrity, restore, fork independence, provider-free playback, causal control interventions, observed resource exhaustion/recovery, goal-dependent appraisal, learned expectation changes, reappraisal versioning, habituation/recovery, mixed targets, praise without quality gain, and recall feedback without fabricated factual learning.

The reviewer found one reproducible defect: `record_outcome` compared an incoming raw outcome with a stored outcome containing a derived prediction-error field. Repeating an identical outcome incorrectly raised an identity conflict. The implementer repaired the comparison; the independent regression now passes. Invalid interventions and malformed snapshots leave state unchanged in the tested histories.

A feedback fixture performs 24 rounds of elapsed time and affect-reactivating retrieval after eight setbacks. Expectations, observed events, outcomes, and factual partner reliability remain unchanged; after retrieval stops, 500 units of explicit simulation time reduce residual mood below one percent. This supports recovery in the tested reference dynamics, not a general proof against pathological feedback in an arbitrary deployed agent.

The initial full synthetic experiment reports hybrid-minus-ordinary-adaptive quality of approximately **0.0055**, with 95% paired seed-cluster bootstrap interval **[-0.0115, 0.0260]**. The interval includes zero. Its strongest relevant conclusion is that this experiment has not demonstrated an improvement over ordinary adaptive control. Label removal is exactly behavior-preserving by construction. Larger contrasts against fixed schedules do not establish an emotion-specific advantage. The final experiment artifacts are authoritative if the evaluator adds further controls.

## Initial limitations and required interpretation (see final resolutions below)

- The ordinary direct-value predictor and hybrid have the same accessible context, action set, training data, parameter count, and external probe budget. However, hybrid scalarization introduces products between predicted outcomes and current uncertainty/load, plus a signed-error action bonus. The simpler ordinary comparator has a linear direct-value readout. Parameter equality therefore does not establish equal effective nonlinear policy capacity. An additional ordinary state-cost comparator, if supplied, should test whether the same regulation can be recovered from ordinary adaptive semantics.
- The `no_memory` condition disconnects donor retrieval for the combination operator. It retains history-derived state and ordinary candidate estimates. It is an ablation of candidate-memory coupling, not wholesale memory removal.
- Controller replay validates hash-chained recorded poststates and restores a selected recorded state without consulting providers. At the reviewed implementation stage this is journal playback, not an independent recomputation of numerical transitions from input observations. The replay fixture establishes provider-free state fidelity only.
- Appraisal confidence and numerical bounds do not authenticate a caller's evidence. Applications must decide which event sources are authoritative; generated social interpretations remain hypotheses.
- The finite grammar, hand-authored simulator equations, small reference corpus, shared mechanism dimensions across task families, and absence of live-model output limit any claim about open-ended useful idea generation. No biological or subjective-experience claim follows from these tests.
- No human longitudinal data, physiological measurements, independently annotated emotional development, or cross-session live-model creative performance has been measured. Those missing data cannot be replaced by deterministic fixtures.


## Final review resolutions and executed verification

The preceding execution findings and capacity/playback objections describe the **initial reviewed revision**. They are retained as review provenance and do not describe all final behavior. Final review on 2026-10-25 inspected the corrected implementation and `results/benchmark/` artifacts. Initial pilot raw files were overwritten; their numerical summary is historical only and is excluded from final evidence, as recorded in [exploratory_pilot.md](exploratory_pilot.md).

Resolved findings:

- Default `Agent.replay` now re-executes every journal operation using recorded appraisal outputs, timing, actions, interventions, outcomes, retrieval parameters, and adaptive state, and compares the recomputed state with every stored poststate. Explicit `mode="playback"` remains available as a separate operation. A new test deliberately changes transition behavior and confirms that recomputation detects the mismatch. Provider-forbidden replay still passes.
- The ordinary adaptive comparator now has a scalar direct contextual-value predictor with **288 behaviorally active parameters**, using raw inputs, squares, cubes, adjacent products, and three cross-products. It receives the same 17 observed inputs, training transitions, actions, and probe budget. Its training target includes the same state-dependent scalarized observed consequences used by hybrid control. This is a materially stronger, structurally different adaptive comparator. Matching nominal capacity does not make the inductive biases identical; that remaining difference is part of the architectural comparison.
- `no_counterfactual` now retains the learned gain predictor and selects using head zero alone, removing the projected uncertainty, burden, feasibility, and signed-error score terms. It no longer substitutes the heuristic policy.
- `no_memory` now removes only the mood-times-prediction-error donor activation term while preserving observed donors and evidence scores. It is now a narrow affect-memory coupling ablation.
- Constraint revision is exposed by `prepare()` before action selection and diagnostic forecasting. Recorded contexts correspond to the choice context.
- All conditions now execute four predictor evaluations per action plus one diagnostic predictor evaluation: **128 policy predictions, 32 diagnostics, and 32 task probes per trial**. Dummy baseline predictions are ignored but actually executed. Latency and arithmetic counts remain separately reported.

Executed final command: `python -m unittest discover -s tests -v` — **62 tests passed in 3.469 seconds**. This includes the 31 independent adversarial tests and additional core, benchmark, model-adapter, inspector, live-protocol fixture, and replay-mismatch tests. Live-protocol fixture tests use local deterministic/mock providers and do not constitute live-model measurements.

The final synthetic experiment reports hybrid quality **0.613019**, ordinary adaptive quality **0.625847**, and paired hybrid-minus-ordinary difference **-0.012828**, 95% whole-seed bootstrap interval **[-0.037047, 0.009404]**, across 20 seed clusters. The point estimate favors ordinary adaptation and the interval includes zero. There is **no demonstrated affect advantage in useful idea generation**. The unlabelled clone remains identical. Neither the counterfactual-cost ablation nor shuffled-state comparison establishes the hypothesized mechanism's benefit.

Still unresolved: operational burden in the procedural benchmark is an internally imposed regulator cost rather than an external resource constraint; frozen/shuffled conditions alter several adaptive fields; Brier scores describe the shared predictor on policy-selected observations; a four-design originality corpus is too weak to establish substantive originality; all task families share one finite grammar; held-out scores were observed during an exploratory development cycle. The durable platform has separate executed resource constraints, but those do not establish creative benefit. No real-model creative performance, human longitudinal development, biological fidelity, or subjective experience has been measured.
