# Independent experiment audit

**Historical audit:** Findings in the initial sections concern the first implementation. The final resolution section below identifies corrections and the authoritative final results. Initial raw pilot files were overwritten and are not final evidence.

Reviewed 2026-10-25: `regesh/benchmark.py`, `regesh/learning.py`, fixed configuration, and the first executed benchmark artifacts. The reviewer did not edit the benchmark or select its results. Findings apply to that reviewed revision; later corrections should be identified in the final report. The initial artifacts are evidence of an executed procedural controller experiment, not live-model creativity.

## What is sound

- Each default test trial starts with a fresh `SearchEnvironment` and `SearchState`; shared trained weights remain fixed. Default seed partitions do not overlap, and the scientific formula family is absent from training.
- Candidate random bits are coupled by task seed, step, and action. Conditions receive the same four operators, probe stream, reference designs, and 32 external probe calls. Diagnostics make no extra oracle calls; predictor arithmetic is counted separately.
- Training action selection is randomized, so all four heads receive outcome examples rather than only observations selected by the final policy. The two learned models use the same 6,144 training transitions.
- Final adjudication uses a separately seeded 64-scenario set without a language judge. Pairing and whole-seed bootstrap clustering account for dependence between the three task families sharing a seed. Twenty seed clusters are still a modest uncertainty sample.
- The report explicitly limits transfer to formula changes within a shared six-coordinate grammar and disclaims live-model/published novelty evidence. The unlabelled hybrid produces identical final designs in every reviewed trial, as expected.

## Concrete validity issues in the first run

1. **The ordinary comparison matches stored size, not active behavioral capacity.** Both predictors store 288 coefficients, but only ordinary head 0 affects actions: 72 coefficients are behaviorally active. The hybrid uses every head with additional products between context and predicted consequences. The remaining ordinary heads are auxiliary and unused by its policy. Matching array size does not establish matched capacity.

   This is stronger than a hypothetical concern: on the saved weights, ordinary head 0 equals `hybrid_head0 + .03*hybrid_head1 - .02*hybrid_head2 + .04*hybrid_head3` with maximum absolute residual **5.39e-15**. Linear ridge fitting commutes with the fixed target combination. Thus this baseline is a fixed-weight reduction of the same transition predictor, rather than a strong structurally different adaptive policy. Add a direct-value policy with active interaction features and similar trainable capacity, or report this limitation prominently. Keep the same-mechanism no-label clone separately.

2. **`no_counterfactual` is exactly the heuristic policy.** All 60 reviewed final designs match the heuristic. This removes the entire learned policy, not solely projected consequences. It cannot attribute a difference specifically to counterfactual regulation. A useful ablation would retain the same learned quality predictor and other controls while removing only predicted uncertainty/burden terms or their contextual weighting.

3. **`no_memory` is not disconnected affect-memory coupling.** It changes `combine`'s donor from observed candidates to the static corpus; memory selection in the main benchmark is not weighted by affect. This tests experiential donor access. Keep that interpretation, and independently test affect-biased versus evidence-only retrieval in the platform. Do not claim that this ablation measures affect-memory feedback.

4. **The benchmark's operational burden is a control signal without an independently constraining consequence.** `load` changes scores and is reduced by verification but does not change design feasibility, time budget, available operators, error rate, or probe cost. Its regulation therefore need not be useful for a reason external to the controller's objective. Platform operational constraints can be tested separately. A stronger environment would make accumulated work consume a real budget or change task opportunities, equally for every controller.

5. **Revision occurs after the policy has selected its boundary-step action.** At `step == budget//2`, the policy and diagnostic forecast see the old phase/context, then `env.step` changes the phase and clears estimates before constructing the training/trace context. The recorded transition context therefore differs from the context that selected the action on that boundary. Either expose the constraint change before action selection or record both contexts and describe the one-turn observation delay. This matters for exact policy replay and causal trace interpretation.

6. **“Original” is nearly vacuous here.** The corpus has four of 4,096 configurations, so 99.90% of uniformly sampled configurations are absent. Every reviewed hybrid and ordinary output is benchmark-original. Useful-original fraction mostly tests the quality/compliance thresholds, not originality. `mechanism_diversity` counts distinct parameter configurations; it does not independently identify different causal mechanisms. These are valid finite-grammar diagnostics with narrow labels, but they are not evidence of original engineering ideas or a broad creativity result.

7. **The Brier score uses the hybrid predictor for every condition.** It measures calibration of one shared outcome predictor under different policy-induced observation distributions. It does not compare each condition's own confidence or factual support. Rename/explain the metric, or add condition-specific forecasts with equal information and training budgets.

8. **Frozen and shuffled conditions are broad interventions.** Frozen disables outcome expectations, failures, progress, and burden adaptation as well as mood. Shuffled samples prior regulation tuples with replacement; it does not preserve the realized trial marginal exactly. Both remain causal manipulations of this state, but neither isolates emotion from ordinary history statistics. Report exactly which fields are altered and avoid calling the frozen difference an emotion-specific effect.

9. **External search budget is matched; total computation is not fully matched.** Learned policies perform four candidate predictions per step, while simple baselines do not; all conditions additionally perform one shared diagnostic prediction per step. The actual counts are recorded, which is good. Describe the experiment as equal oracle/candidate budget, report arithmetic/latency, and use the matched learned comparator for the strongest claim. The 64 pre/post adjudications are evaluation-only, unavailable to policies.

10. **Initial state and provenance are reconstructable but not self-contained snapshots.** `initial_snapshot` stores `SearchState` while task seed/config, environment phase, generic weights, and policy state live elsewhere. The row's family/seed and saved configuration reconstruct the default beginning, but the ordinary weight artifact should also be hashed. Report a source revision, NumPy/Python versions, and raw-data hashes for a stronger reproducibility record.

## What the first artifacts actually show

The initial hybrid mean quality is **0.6324**, ordinary **0.6269**, paired difference **+0.0055**, 95% seed-cluster interval **[-0.0115, +0.0260]**. This is inconclusive against even the initially weak ordinary comparator. The hybrid improves over some fixed/heuristic schedules under this procedural environment; shuffled state and experiential-donor removal have intervals spanning zero. These results support execution and some adaptive-control effects, not affect-specific superiority. If reruns follow substantive corrections, use their measurements instead and retain this paragraph as a dated audit of the first run or explicitly supersede it.

The primary question “does affect improve useful idea generation?” remains **unestablished**. No real language model generated or revised solutions in this benchmark, and its originality reference is too small to discriminate mechanisms. A negative/inconclusive answer is scientifically acceptable and should not be obscured by favorable comparisons to static schedules.

## Highest-value correction and next experiment

First strengthen the ordinary adaptive policy using all of its matched capacity and correct the mechanism ablation. Then run a preregistered held-out test with a real, independently constraining resource budget. For live-model work, use the same model, prompts, information, proposal/check budget, and initial persistent snapshot across hybrid and ordinary control; adjudicate executable constraints blind to condition and assess originality against a substantive retrieved reference corpus. Record every appraisal/reflection call against the same budget. A larger fixed seed set alone will not repair the current validity issues.

## Final independent review resolutions — 2026-10-25

A second reviewer inspected the corrected code and final `results/benchmark/` artifacts. The following changes resolve concrete issues raised above without erasing their provenance:

| Initial finding | Final disposition |
|---|---|
| Only 72 ordinary parameters affect actions | Replaced by a structurally different scalar direct-value model with 288 active nonlinear-feature parameters; same 17 observations and same state-dependent observed-return objective |
| No-counterfactual equals heuristic | Corrected: learned outcome head zero remains; projected non-quality terms are disconnected |
| No-memory removes experienced donors | Corrected: only mood-times-prediction-error donor ranking is removed, preserving all experienced donors |
| Boundary revision context mismatch | Corrected: `prepare()` reveals the revision before policy choice and forecasts |
| Unmatched predictor arithmetic | Corrected: each condition actually executes 128 policy prediction calls and 32 diagnostic calls per 32-probe trial |
| Diversity counts only canonical parameter configurations | Both canonical diversity and included-mechanism-subset diversity are reported; neither establishes open-ended originality |
| Playback only | Platform default replay now recomputes all operations from recorded provider outputs and checks each poststate; a deliberately patched transition is detected by tests |

The ordinary comparator's polynomial feature basis and the hybrid's predicted-consequence factorization still differ in inductive bias. Parameter-count matching is a useful control, not a proof of identical functional expressiveness. Both see the same information and optimize the same scalarized observed consequences. This supports a substantially stronger architectural comparison than the pilot.

The corrected study trains on 6,144 transitions and evaluates 660 held-out-condition trials (20 seed clusters, three formula families, 11 conditions), with 88 development-condition trials. Final hybrid quality is **0.6130188** versus ordinary adaptive **0.6258467**. Paired difference is **-0.0128279**, 95% seed-cluster bootstrap interval **[-0.0370468, 0.0094043]**. The result does not demonstrate an advantage for affect; the point estimate favors ordinary control. Label removal leaves results unchanged. Removing projected non-quality costs produces quality **0.6311**, further failing to support the proposed benefit of those terms.

Remaining concerns 4, 6, 7, 8, and 10 retain force in their stated scope: internal burden is not an external task constraint; corpus absence is weak originality evidence; shared-predictor Brier does not compare independent policy confidence; frozen/shuffled interventions are broad; complete trial provenance requires configuration, task identity, weights, and code together. The pilot informed structural corrections after scores were visible, so the final study remains **exploratory**, not a preregistered confirmation. The pilot raw files were accidentally overwritten and are not claimed as reproducible evidence; see [exploratory_pilot.md](exploratory_pilot.md).

Final verification: `python -m unittest discover -s tests -v` completed **62 tests, all passing**, including 31 independent adversarial history/invariant checks. No live-model or human creativity measurements are included in those tests. The next informative experiment remains a fresh, prospectively fixed comparison against this stronger adaptive baseline, using actual executable tasks, externally constraining resources, and a substantive prior-art corpus, followed by equal-call live-model evaluation when an endpoint is available.
