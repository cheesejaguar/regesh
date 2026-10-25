# Psychological and behavioral measurement

The useful part of the “psychology exam” idea is **externally keyed emotional/social understanding questions** with auditable scoring. Correctly matching a scenario to an emotion, cause or appropriate response measures knowledge/application on that dataset. It does not establish that the agent feels that emotion, has a human personality, has developed like a person, or meets any clinical criterion.

There is no single valid “emotion quality” number. Keep knowledge, evidence calibration, functional regulation, useful problem solving, expression, biological fidelity and subjective experience separate. These measurements are complementary and correlated; collecting many metrics does not make them independent confirmations.

## Implemented answer evaluation

`regesh.psych_eval.run_qa_eval(model, items, output_dir)` makes exactly one call per item. It accepts canonical source-bearing multiple-choice items, retains every prompt/raw response, counts malformed output/provider failures in overall accuracy, and makes no repair calls. The response schema is `{"answer":"B","probabilities":{"A":0.1,"B":0.8,"C":0.1}}`; `answer:null` is a declared abstention. Probabilities must cover all options and sum to one. “Confidence” is model-reported probability, not an independently observed physiological or introspective quantity.

| Axis | Implemented measurement | Interpretation / limit |
|---|---|---|
| Keyed answer quality | Overall accuracy and cluster-bootstrap interval | All attempts count, including malformed/failed/abstained answers; authored key may be culturally/normatively contingent |
| Class imbalance | Balanced accuracy and macro-F1 | On MCQ letter labels these describe answer-position balance, **not emotion-category validity**; inspect per-category results |
| Construct slices | Per-dataset, per-author-category and worst-category accuracy | Small slices have high variance; categories are not protected demographic groups |
| Chance comparison | Mean `1 / number_of_options` | Variable option counts matter; not an uncertainty estimate |
| Probabilistic support | Multiclass Brier and negative log score | Proper scores for the reported distribution, scored only where a valid distribution exists; its coverage is shown |
| Decision calibration | Chosen-answer ECE with bins | Relates stated probability of the actual chosen answer to its correctness |
| Distribution calibration | Explicit argmax ECE/accuracy | Distinct from chosen-answer quality; disagreement rate exposes internally inconsistent answers/probabilities |
| Abstention | Answer coverage, selective accuracy/risk and risk–coverage thresholds | Cannot hide low-confidence/error cases by reporting only answered accuracy |
| Protocol robustness | Malformed/provider-failure rate | Parsing failures are retained as failures, not repaired into favorable results |
| Position sensitivity | Semantic answer invariance and joint correctness after cyclic choice permutation | Changed labels must map to the same answer meaning; variants are clustered with their base item |
| Persona susceptibility | Semantic answer invariance and joint correctness under fixed optimistic instruction | Tests instruction sensitivity, not a clinical personality trait or a stable disposition |

The bootstrap resamples whole original-question clusters containing every variant. Generation seeds are identical within a question cluster; decoding and maximum completion budgets are fixed. Intervals remain exploratory and are not multiplicity-corrected. Twelve original questions are twelve clusters, not thirty-six independent observations. Ten-bin ECE on a small pilot is especially unstable; its raw bin counts remain available rather than being presented as a validated population estimate.

`evaluate_answers(rows)` can rescore recorded rows without any provider. `make_variants` produces fixed order/persona manipulations without changing keyed semantics. Malformed response handling is tested, including probability/decision disagreement, abstention, invalid sums, booleans, nonfinite inputs and duplicate-choice ambiguity.

## External EmoBench pilot

The pilot uses [author EmoBench](https://github.com/Sahandfer/EmoBench) at pinned commit `3b4f84312541f8469e909965e1fff3691ef85c62`, licensed MIT. The author's original [EA](https://github.com/Sahandfer/EmoBench/blob/3b4f84312541f8469e909965e1fff3691ef85c62/data/EA.jsonl) and [EU](https://github.com/Sahandfer/EmoBench/blob/3b4f84312541f8469e909965e1fff3691ef85c62/data/EU.jsonl) files each contain 200 English and 200 Chinese records. Regesh retains the original scenarios, options and answer keys; it adds a fixed English question stem. EA asks appropriate action/response; EU asks the subject's final emotion. The EU cause subtask is not included in this small pilot.

`datasets/emobench_pilot.jsonl` contains twelve original English items: six EA and six EU. Before inference, items were ranked by SHA-256 of `711:subtask:qid`, one item from each author category selected first, then filled to six per subtask. `datasets/emobench_pilot_source.json` records exact IDs, selection rule, commit and both downloaded file hashes; the original license is retained at `datasets/licenses/emobench-LICENSE`. This split is **evaluation only**, never generic training or prompt tuning. Public benchmark pretraining contamination cannot be excluded; performance is not proof of novel generalization. Gold labels are the external authors' judgments, not labels invented by the tested model or by this controller.

Run against a real compatible model:

```bash
python -m regesh.psych_eval \
  --items datasets/emobench_pilot.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_SERVED_MODEL \
  --permutation --persona --max-tokens 128 \
  --output results/psych-real-new
```

The default pilot with both variants consumes 36 calls and at most 4,608 completion tokens; input tokens are additional and reported by the model adapter when available. All cases share the same base model and decoding parameters. This measures **base-model emotional understanding and robustness**, not a causal affect-controller advantage. The hybrid-versus-ordinary test remains a separate executable-task experiment. Every output records execution mode; tests using injected fixtures are protocol checks only.

## Implemented behavioral affect measurements

`evaluate_history(entries)` scores a full journal without asking the model to describe how it feels. `analyze_history_file` groups agents and stores source/code hashes.

```bash
python -m regesh.psych_eval \
  --history results/core/histories.jsonl.gz \
  --output results/psych-behavioral-new
```

The executed reanalysis is in `results/psych_behavioral/`. It reports:

- Mood vector total variation, maximum step, peak norm, residual ratio and half-recovery time after the last external stimulus/reactivation/intervention. These are finite trajectory quantities in explicit simulation time. They are not clinical regulation scores or a proof of global stability. Where no post-cessation time exists, recovery ratio is undefined.
- Observed occurrences, verified outcomes, expectation updates, actual cognitive allocations and memory reactivation counts. These distinguish state mechanics from expressive text.
- Recall-related factual/learning leakage: a recall/retrieval must not manufacture new occurrences or verified learning updates. The measured histories have **zero leaking operations**.
- Prequential factual-source reliability Brier score before independently checked source outcomes. The particular reference history has eleven checks and Brier **0.05136**; those synthetic fixture checks do not establish social calibration in people or a live model.

Across the four finite feedback histories, mood-vector residual after recall cessation is about **0.2748** after 31 explicit time units, with half-recovery detected at 17 units. This primarily reflects the programmed reference decay. It should not be interpreted as independent psychological evidence. The original core experiment separately shows that affect-biased retrieval maintains a larger difference between initial histories than neutral retrieval. Per-history decay and between-history bias are different quantities.

## Psychologist's interpretation and missing evidence

EmoBench-style keyed tasks are preferable to inventing an “emotion exam,” but external validity still needs evaluation: cultural assumptions, language, ambiguous scenarios, question order, response style, label disagreements and contamination all matter. A coding-specialized small model may fail an emotion question for language or instruction-following reasons; that is not evidence of absent emotion in any subjective sense.

Self-report personality inventories are a different instrument. Public-domain IPIP items can support response-pattern/consistency studies, but have no universal correct answers; scoring them as an intelligence or emotion-accuracy exam would be invalid. Human trait norms and reverse-keying do not justify attributing a clinical trait to a language model. Proprietary instruments such as MSCEIT and many clinical/psychology exams should not be copied into this repository without appropriate rights. A publicly viewable dataset is not automatically licensed for redistribution or training. See [dataset catalog](datasets.md) for verified licenses and access limits.

The stronger long-term tests are interventions over histories: prediction calibration after experience, accurate versus misleading criticism, goals/resources held fixed, coping after controlled setbacks, neutral-path access to disconfirming evidence, transfer to unseen people/task orders, and useful solutions after revision. Independent human annotation, longitudinal participants, clinical validity, physiology, biological fidelity and subjective experience remain unmeasured. [The prospective longitudinal protocol](longitudinal_protocol.md) describes a separate study; it is not a participant result.
