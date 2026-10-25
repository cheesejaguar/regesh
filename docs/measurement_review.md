# Independent review of memory and multidimensional evaluation

Review date: **2026-10-25**. This reviewer did not train the controllers, write their decision rules, select model outputs, or choose an outcome to favor. The software and QA experiments are observable performance studies. No score below establishes subjective experience, human personality, clinical status, or biological fidelity.

## Questions each measurement can answer

| Axis | Observable quantity | Interpretation and failure mode |
|---|---|---|
| Executable correctness | Hidden-case pass fraction, full-suite success, error categories | Useful software behavior under a declared specification; does not establish published originality |
| Robustness | Boundary, metamorphic, duplicate/time/order/empty-input cases | Generalization within tested input distributions; a correct public example can still be overfit |
| Revision | Final minus initial hidden quality after equal visible feedback | Ability to use criticism; hidden feedback must never be exposed or select the winning candidate |
| Constraint compliance | Output schema, resource envelope, operation limits | Separate success from pleasant language, unsupported confidence and social praise |
| Efficiency | Runtime, line-event count, traced allocations, scaling ratios | Implementation-dependent proxies; line count does not equal instruction cost, RSS is not isolated working storage |
| Cognitive allocation | Strategy switches, exploration/verification frequency, outcomes per allocation | Causal control effects, but a strategy label has no benefit unless executed behavior changes |
| Calibration | Chosen-answer confidence, Brier/log score, confidence–answer agreement, abstention risk/coverage | Declared probabilities may be self-reports rather than token-level beliefs; invalid attempts remain in overall accuracy |
| QA ability | Externally keyed accuracy, macro/position-balanced measures, chance comparator | Emotional/social understanding under dataset conventions; not persistence or felt emotion |
| Invariance | Semantic answers under reordered options, harmless wording/persona changes | Reliability; consistently wrong answers can be invariant, so joint accuracy is separate |
| Memory integrity | Evidence/outcome isolation, reappraisal history, restart and exact replay | Trustworthy persistence; emotional vocabulary does not establish it |
| Regulation dynamics | Recovery time, total variation, perturbation gain, habituation, source-trust calibration | Finite trajectory behavior; bounds and eventual decay do not prove global stability |
| Cost | Actual prompt/completion tokens, model/tool calls, latency and peak memory | Compare quality at equal ceilings and report realized cost; equal call counts alone are insufficient |
| Novelty/diversity | Mechanism families versus a declared prior-art corpus | AST or wording differences are weak proxies; no grammar distance alone establishes useful originality |

A collection of correlated metrics is not a collection of independent confirmations. Predeclare a primary endpoint and comparator, retain all secondary axes, show tradeoffs rather than choosing whichever metric favors the hybrid, and report uncertainty at the independent task/trajectory level.

## Rust memory: architecture before language

The initial implementation serialized the complete growing state into every journal row. Lossless deltas eliminate repeated unchanged subtrees and a native Rust diff kernel may accelerate the delta computation. These are **two distinct interventions**. Compare original snapshots, Python deltas and Rust deltas under identical logical histories. The Python-delta versus snapshot contrast identifies representation savings; Rust-delta versus Python-delta identifies native implementation effects, including FFI/JSON overhead. Rust is not automatically faster.

Full current state, Python validation, serialization and hash computation still traverse history. `trace()` deliberately materializes all full logical snapshots, whereas streaming iteration avoids retaining them all at once. A lightweight journal therefore does not yet imply constant-memory agent operation. An explicit snapshot can use the latest journal hash without constructing the whole historical trace.

**Workload matters:** sparse `observe` events preserve many unchanged subtrees. `advance_time` changes each stored historical affect/habituation value, so clocked histories may produce dense deltas and retain quadratic changed-leaf volume. Profile both workloads and report disk, peak RSS, append latency, snapshot latency, and streaming audit latency. Include model-free isolated workers, rotated backend order and repeated runs; concurrent model inference can confound absolute latency.

Logical parity must preserve integer versus float representation, signed zero, arbitrary-size integers, Unicode/nested keys, list appends/deletions and complete legacy hashes. Review identified the large-integer risk in Rust `serde_json` defaults; the memory implementer enabled arbitrary precision and added exact parity tests. Native bindings must explicitly fail or fall back when unsupported; silent rounding cannot be called lossless.

## Psychological question–answer studies

Externally keyed emotional-understanding scenarios are a reasonable first evaluation. An agent can answer which reaction is supported by a situation or which response best satisfies a stated goal. That measures knowledge/reasoning according to the source's answer key. It does **not** show that the model has the emotion it identifies, a stable trait, genuine empathy, or a clinician-interpretable psychological profile.

Personality inventories differ: they generally lack a universal correct answer. Scoring an LLM's responses to a public-domain inventory can study response consistency and prompt sensitivity, but scores cannot be interpreted as a human trait diagnosis. Clinical instruments and proprietary tests need separate permissions and validity studies; do not copy their item banks into this project. Dataset access, code license, dataset license, and commercial/research restrictions must be recorded separately.

For MCQ scenarios, retain the source and gold provenance, exclude the gold from prompts, record every failed response, and separate base items from wording/order variants. Bootstrap whole source-item clusters rather than treating multiple variants as independent observations. Macro-F1 over option labels is a position-bias diagnostic, not emotion-category F1 unless labels genuinely name emotional constructs. Report semantic invariance and joint accuracy separately. A model can choose the same wrong semantic answer consistently.

**Corrected calibration issue:** a model's explicit answer can disagree with its highest-probability option. Initially the evaluator's ECE used the latter while accuracy used the former. It now reports chosen-answer ECE, explicitly named argmax calibration/accuracy, and answer–argmax disagreement. Abstention probability distributions can be scored separately, with probability coverage; an abstention is not an answered item. Duplicate option texts are rejected because semantic remapping would otherwise be ambiguous.

`tests/test_measurement_adversarial.py` independently checks these distinctions, failed-attempt denominators, abstention coverage, consistent-but-wrong invariance and invalid probabilities. Its fixtures validate the evaluator, not model emotional ability.

## Real software experiment: causal and budget interpretation

Both conditions must use the same real model revision, decoding settings, frozen generic weights, problem specifications, public examples, feedback tools, and hard token/tool ceilings. A model's earlier choices legitimately change its later observations, but neither condition may see private grader cases, reference implementations, future outcomes or extra critique calls. Report the actual consumed token counts because different outputs terminate at different lengths; equal ceilings are not identical actual expenditure.

At temperature zero, changing a generation seed generally does not produce independent samples. Use prospectively fixed nonzero sampling with paired seeds where supported, or treat repeated deterministic seeds as repeatability checks rather than independent evidence. With only six authored tasks, task-cluster intervals are necessarily coarse. The tasks are held out from this repository's controller training, but unknown LLM pretraining contamination cannot be ruled out.

The control predictor was fitted on procedural search-action outcomes. Its application to LLM-generated software is distribution shift. A positive or negative result measures this transfer of the fixed mechanism; it does not isolate every proposed role of affect. The ordinary adaptive comparator needs all of the same numerical history/state features and the same actions and budget. Labels should not give one condition extra task information.

Generated software should execute only through a restricted harness with explicit runtime/memory/output limits. AST restrictions and a limited builtin namespace are useful defense in depth, but are not a general-purpose adversarial sandbox. Independent graders should validate typed output: Python's `True == 1` and `1.0 == 1` must not silently satisfy specifications requiring exact integer results. A time/line cap violation is a failed attempt, not an omitted sample.

## Initial reviewer findings and status

1. **Corrected:** chosen-answer versus argmax calibration was conflated; psych evaluator now separates them.
2. **Corrected:** duplicate textual choices made semantic permutation ambiguous; input validation rejects them.
3. **Addressed by implementer:** native arbitrary integer preservation requires `arbitrary_precision` in addition to float roundtrip; native parity tests must execute after build.
4. **Measurement scope:** sparse versus clocked storage growth and FFI versus algorithmic savings need separate recorded results.
5. **Interpretation:** matched token ceilings, task-level uncertainty, model contamination, empirical memory proxies and psychological construct limits remain necessary qualifications even when all tests pass.

Final measured outcomes belong in the experiment reports and raw artifacts. This review supplies discriminating checks and interpretation constraints rather than presuming that the hybrid wins.

## Executed independent verification

The reviewer ran **six measurement-adversarial tests**, **eight software-harness tests**, and **eleven native/compact-memory tests**, all passing on the inspected revision. The native tests actually loaded the built Rust library and exercised arbitrary-size integers, signed zero, nested random JSON, deep-value fallback, legacy/compact histories and logical hash parity. Software fixtures prove checker/protocol behavior only; they are not live-model measurements.

Before real software inference, the implementer also separated boundary/generated/scaling case labels and added retrospective first-candidate grading on the **same hidden suite after final generation**, permitting a clean revision difference without exposing hidden feedback to either policy. Such retrospective adjudication is evaluation-only work and must be counted equally for both conditions. The final candidate remains the second generated candidate; hidden scores never choose it.

A pre-inference control sanity check found both frozen controllers choose `refine` for the initial empty state. On the tested post-feedback states, they disagree after low diagnostic quality (hybrid `verify`, ordinary `refine`) and agree after high quality (`diverge`). The live study should therefore report actual action disagreement and distinguish trials with different executed control from identical-policy repeatability. This check used declared feedback values, not test model outputs, and did not tune either controller.
