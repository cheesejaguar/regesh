# Prospective longitudinal appraisal and useful-idea study

**Status: unexecuted protocol.** No participants, annotators, physiological measurements or human outcome data were available. This document supplies a design for missing external evidence; it contains no participant results and does not propose testing subjective experience by asking a model how it feels.

## Questions and preregistered units

Separate two hypotheses: (A) verified history changes future appraisal/coping in an evidence-calibrated way, and (B) that regulation improves useful ideas beyond ordinary adaptation. Units are independent agent trajectories paired by base model, task and seed. Human judgments are annotations of observable evidence and outputs, not direct access to an artificial or human subjective state. Choose the sample size prospectively from a power simulation targeting a declared smallest useful improvement; do not choose it after viewing effects. Declare model versions, task corpus, partitions, primary endpoints, multiple-comparison policy, exclusion rules, costs and missing-response handling before collection.

## Data collection and partitions

Build a reviewed corpus of longitudinal histories with four sessions per trajectory: initial collaboration, supported setback, correction/new context, and a later structurally related task. Include success without praise, praise without success, ambiguous intent with later resolving evidence, helpful accurate criticism, unsupported criticism, source-reliability reversal, heavy but controllable work, resource shortage, recall and quoted failures. Keep the same observation under different accepted goals/internal-resource histories. Identify every observed claim, confidence, checker result, alternative interpretation and causal link. Do not train on annotators' invented facts or LLM-generated emotion labels.

Have at least two independent annotators mark evidence support, plausible interpretations, goal relevance, uncertainty and feasible coping options; adjudicate disagreements while retaining the original annotations. Assess inter-annotator agreement separately for factual support and interpretation. Split **people/role identities, task families, context changes and event orders**, not just individual utterances. Quarantine held-out histories and prohibit prompt tuning after examination. All manipulations should be explicitly simulated/role-played scenarios; any real participant recruitment and data collection needs its own reviewed consent/privacy procedure before it begins.

## Matched experimental conditions

Compare the same base model and information access under memory/planning, static persona, turn-local conditioning, persistent heuristic, learned hybrid, and structurally different ordinary adaptive regulation. Include exact hybrid label removal, frozen state, within-history temporal misalignment, disconnected affect-memory ranking and internal-projection removal. Use complete declared initial snapshots including trained-weight identifiers, memories, source beliefs, goals, time, PRNG state and resource budget. Shared weights are frozen; only declared within-trajectory adaptation occurs. Count every appraisal/reflection call, input/output token, retrieval, tool check and wall-time cost. Give ordinary adaptation every numerical history feature available to hybrid. Randomize condition execution order to reduce server/version drift.

## Independent endpoints

For objective A, test counterfactual probes with identical new evidence after different verified histories. Score factual calibration, supported reappraisal, unsupported-intent promotion, source reliability, stability under sustained perturbation, recovery after cessation, and neutral-path retrieval of disconfirming evidence. Persistent label consistency is secondary and cannot establish functional benefit.

For objective B, use executable software optimization/repair, engineering designs under explicit constraints, distant-domain technique transfer, discriminating experiments for competing explanations, and revision after new constraints. Evaluate goal satisfaction and feasibility with independent tests where possible. Compare mechanism originality against a fixed substantive prior-art corpus. Blind qualified reviewers to conditions and randomize output order for remaining originality/feasibility judgments; require concrete mechanism/evidence rationales, not ratings of unusual wording. Report useful-but-common and original-but-infeasible separately. Reviewers should not generate both the training labels and the final outcome judgments from the same examples.

Primary analysis is paired verified task utility versus ordinary adaptation, with uncertainty clustered by independent trajectory/task rather than treating repeated turns as independent samples. Report family-level transfer, improvement after correction, all failures, annotation disagreement, actual computation, latency and sensitivity to reasonable reference-corpus coverage. Secondary analyses cannot rescue a negative primary comparison. Physiological or consciousness claims require separate evidence and are outside this protocol.

## Decision rule

Evidence of persistence/development without improved task performance supports a working architecture, not objective B. A favorable persona comparison without an advantage over strong ordinary adaptation does not support an affect-specific mechanism. If externally reviewed useful originality fails to improve, retain the negative result and test a narrower causal explanation before proposing more affect complexity.
