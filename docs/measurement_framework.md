# Measuring quality across constructs

The extended study separates engineering efficiency, externally keyed capabilities,
functional regulation, and useful solution generation. None is a direct measure of
subjective experience. A single aggregate score would hide important tradeoffs.
This document identifies the measures implemented and the limits of each.

| Construct | Implemented measurements | Interpretation and limit |
|---|---|---|
| Lightweight persistence | SQLite bytes including checkpointed journal; logical payload bytes; bytes/event; growth over history lengths; native library size | Storage overhead, not an affect-quality score. Sparse and clocked histories differ. |
| Runtime cost | End-to-end write latency; per-write p50/p95; isolated process peak RSS; snapshot/stream/replay costs; matched backend runs | Separate compact-representation gains from Rust-vs-Python gains. Shared-host contention limits precision. |
| Persistence correctness | Same state/hash after restart, mixed legacy/compact replay, snapshot/fork isolation, duplicate rejection | Mechanical evidence of durability, not human memory validity. |
| Executable usefulness | Hidden-case pass fraction, all-tests-pass fraction, strict output contracts, exception/format rates | Independent oracle adjudication; no reward for a model claiming success. Authored tasks remain a small test universe. |
| Constraint compliance | Restricted-source validation, process/line/memory limits, returned-output contracts | Execution compliance. It does not prove an asymptotic memory bound or sandbox arbitrary hostile Python. |
| Generalization/robustness | Boundary cases, independently generated cases, metamorphic transformations, larger-input cases | Category membership follows test construction. Tests never fully prove a specification. |
| Revision after criticism | First and final candidates scored on the **same hidden suite**, both after generation completes | Clean within-trial revision measurement; hidden feedback is not revealed or used to select candidates. |
| Efficiency of correct solutions | Traced line events, fourfold-input cost ratio, peak traced allocation, latency, token/tool usage | Report only alongside correctness. Tracing/output allocation overhead is included; fast wrong code is not useful efficiency. |
| Cost-benefit tradeoff | Verified pass rate per reported token/call, completion ceilings versus actual usage, condition cost tables | Equal token ceilings do not guarantee identical realized usage. No unmeasured energy/carbon claims. |
| Solution diversity | Normalized AST identity and observed operation-method signatures | Syntactic/data-operation proxies, not proof of mechanism originality or research novelty. |
| Calibration of code claims | Optional probability all tests pass, Brier score and confidence coverage | Missing confidence stays missing. Model self-report is a prediction to check, not subjective certainty. |
| Emotional/social understanding | Externally authored keyed situational questions, exact accuracy, balanced accuracy, macro-F1, category slices | Measures answering these questions. Multiple-choice position labels are not psychological classes; class-balanced scores are diagnostic only. |
| Proper probabilistic scoring | Multiclass Brier score, clipped log loss, explicit-answer and argmax ECE, disagreement rate | Coherent answer distributions can still be wrong; calibration excludes malformed distributions and always reports coverage. Small-sample ECE is bin-sensitive. |
| Abstention/reliability | Answer/probability coverage, selective accuracy/risk, threshold risk–coverage curve | Prevents apparent accuracy from silently discarding failures. Abstention is not introspective consciousness. |
| Response robustness | Option-order semantic invariance, jointly correct paired responses, optimistic-persona susceptibility | A consistently wrong model can be invariant. Pair completeness and joint accuracy must accompany consistency. |
| Appraisal development | Identical later probe after verified history; expectation/source reliability changes; old evidence preserved on reappraisal | Shows programmed adaptation. External emotional-validity annotations would be a separate study. |
| Affect-memory feedback | Mood perturbations, selective retrieval, factual-evidence invariance, separation during feedback and after cessation | Finite trajectories can reveal bias persistence; numeric bounds are not a global stability proof. |
| Repetition/recovery | Repeated-signal response ratios, recovery half-time after cessation, mixed-target histories | Reference dynamics and scenario-specific observations, not fitted human physiology. |
| Social evidence discipline | Praise/success separation, factual source-check Brier, accurate/misleading criticism with objective quality fixed | Prevents affiliation from becoming truth/authority. Fixture truth flags are not human longitudinal evidence. |
| Causal treatment strength | Chosen-strategy traces, paired action-disagreement rate, frozen trained-weight verification | Identical selected strategies provide no effective treatment contrast; a label alone is not a mechanism. |

## What a psychological test can and cannot show

An externally keyed **emotional-understanding or situational-judgment exam** is a
reasonable capability benchmark: ask the question, retain the raw answer and
probabilities, compare with the published key, and test equivalent question forms.
The MIT-author EmoBench pilot is the present concrete implementation. Its key is
a normative author judgment, not infallible emotional truth; culture, language,
ambiguity and possible pretraining exposure matter.

A personality questionnaire such as IPIP is different. Its items normally have
no single correct answer; scoring an LLM against a desired personality would
measure compliance with that scoring goal. Test–retest response consistency may
be observable, but importing human norms or inferring a mental-health diagnosis
would require validation the present study does not supply. Proprietary clinical
or emotional-intelligence exams are not interchangeable with public-domain items.
The [dataset catalog](datasets.md) records these access and licensing distinctions.

Functional affect is assessed through histories and interventions: does verified
experience change later appraisal, memory or action in a useful and evidence-
respecting way? Emotional knowledge, fluent empathy, persistent state, functional
control, development, biological fidelity and subjective experience are separate
claims. The QA test addresses knowledge and response reliability; the controlled
software experiment addresses useful regulation; neither establishes feelings.

## Statistical interpretation

The real software comparison uses paired task/seed trials, but the independent
family units are the six authored tasks. Primary intervals resample task means,
keeping generation seeds within tasks. The psych pilot has twelve original items,
not thirty-six independent questions after order/persona variants. Its intervals
resample original item clusters. All malformed responses, tool failures and null
results remain in denominators. Secondary axes are exploratory and unadjusted
for multiple comparisons; report effect sizes and uncertainty rather than
selecting whichever metric favors the hybrid.

The strongest next validation would add independently authored task families,
substantive prior-art retrieval, blind human originality/usefulness ratings,
longitudinal reviewed appraisal histories and a stronger general model. These
studies must be labeled unexecuted until measured; the present expanded metrics
do not substitute for them.
