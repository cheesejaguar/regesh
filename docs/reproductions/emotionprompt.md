# EmotionPrompt: scoped cross-model replication

Primary source checked **2026-10-25**:
[Li et al., *The Good, The Bad, and Why: Unveiling Emotions in Generative AI*,
ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/li24bs.html).
This peer-reviewed paper extends the differently titled
[2023 EmotionPrompt preprint](https://arxiv.org/abs/2307.11760).
They are not counted as two independent papers. The author project at
[llm-enhance.github.io](https://github.com/llm-enhance/llm-enhance.github.io)
revision `0bc596c5c3a2b14e5abea9062204b27137c4af35` supplies the paper and
illustrations. No experiment source or raw generations were found there; the
occasionally cited `LLM-Enhance/EmotionPrompt` repository returned HTTP404.

We execute the paper's textual intervention on four actual Instruction Induction
execution tasks. This is a **scoped method replication on a different model**,
not an exact reproduction of its aggregate result. It tests static prompt
conditioning; it does not test persistent affect or subjective experience.

## Locked experiment

The published Table5 supplies task instructions; Figure6 supplies stimulus EP02.
We append its career-importance sentence to the unchanged instruction, followed
by the same input and output delimiter in both conditions. The delimiter is a
disclosed reconstruction choice because the original generation code is absent.
There are no examples, tools, appraisal calls, retries, repairs, or model judges.

The original [Instruction Induction repository](https://github.com/orhonovich/instruction-induction),
revision `f7304ae0a35239c9819a08f93fd58426f6b4e4c1`, provides the original
`data/raw/execute/` examples and Apache2.0 license. We select eight of its hundred
examples for each of first-letter extraction, second-letter extraction, addition,
and subtraction, using ascending `SHA256(8121:task:source_id)`. These tasks have
objective short answers and are explicitly part of the paper's experiment.
The four tasks were chosen for feasibility and unambiguous scoring before any
generated answer was inspected, not to maximize an observed effect.

The pinned32-item input, source checksums, license, original metric source,
protocol, and freeze record are in `results/papers/emotionprompt/protocol/` and
`configs/reproductions/emotionprompt.json`. The selected examples were unchanged
except that task instructions and provenance fields were attached. This is an
evaluation subset, not a training split.

Each condition receives32 calls with `temperature=0`,32 maximum completion
tokens/call, the same per-pair decoding seed, and alternating condition order.
The existing real CPU Qwen2.5-Coder-1.5B-Instruct Q4_K_M model is used; its exact
download/server identity is preserved in `results/runtime/model.json`.
Generations are stateless and no result feeds into the next prompt. Actual
prompt/completion tokens, finish reasons, failures, request bodies and raw
responses are saved. Failures count as incorrect. The career sentence adds
prompt tokens: equal generation caps do not make total token counts identical.
There is no length-matched neutral sentence, so any observed difference cannot
uniquely be attributed to emotional semantics. Concurrent CPU experiments make
elapsed times descriptive.

## Quantitative comparisons

The primary score reproduces the original benchmark's `evaluate.py`
normalization: first line and pre-period segment, lowercase, punctuation removal,
and its original replacement rules. These rules are retained even where they
are unusual; a stricter case-sensitive, whitespace-stripped exact score is
reported separately. Nine edge cases matched the pinned upstream function.
No free-form answer extraction or LLM judgment is used.

The analysis reports paired improvement/regression counts, exact two-sided
McNemar testing, fixed-task paired item bootstrap intervals, and four-task cluster
bootstrap intervals. The latter are fragile with only four clusters. Per-task
results and both actual token totals are retained. A single generation per input
does not characterize decoding variance.

Published Table7 reports original zero-shot Instruction Induction average
accuracy **0.6283** and EmotionPrompt average **0.6541**, an absolute increase of
**2.58 percentage points**. That aggregates24 tasks, three other models and
multiple stimuli; it is a context reference, **not the same estimand** as our
EP02-only, four-task, smaller-model pilot. Neither the published best-stimulus
score nor the2023preprint headline is substituted for this comparison.

The executed local result is in `results/papers/emotionprompt/live/summary.json`.
Its config, source snapshot, input, raw calls and per-attempt outcomes accompany
the manifest. Exact live inference is not guaranteed to be bitwise reproducible;
the recorded responses allow deterministic score recomputation without another
model call.

### Executed result

All64 planned calls completed with no provider error. Baseline scored19/32
(59.375%) and EP02 scored22/32 (68.75%). The paired difference was **+9.375
percentage points**, with95% fixed-task bootstrap interval **[-3.125,21.875]**
and task-cluster interval **[-9.375,37.5]** percentage points. There were five
improvements and two regressions; exact McNemar **p=0.453125**. This small study
does not provide convincing evidence of an improvement. Strict and normalized
exact scores happened to coincide on these outputs.

| Original task | Baseline | EP02 |
|---|---:|---:|
| First letter |8/8|8/8|
| Second letter |5/8|4/8|
| Addition |6/8|6/8|
| Subtraction |0/8|4/8|

Actual prompt/completion/total tokens were1462/162/1624 baseline and
1718/101/1819 EP02. Baseline had one32-token truncation; all other completions
ended with `stop`. Latency totals were270.37s and209.55s respectively, under
concurrent CPU contention, and are not a controlled speed comparison.
`audit.json` independently checked all64 scores using the unmodified original
evaluator, plus exact prompt/seed/budget/raw-response/token accounting.

**Exploratory diagnosis after outcome inspection:** several failed arithmetic
responses were correct complete equations rather than bare integers. We added a
separately labeled finite parser that accepts only a complete integer or a full
equation with the *original operands*, required operator and arithmetically
correct result. It never executes generated code or guesses from prose.
It found six such correct equations rejected by the primary metric in baseline,
and two in EP02. Arithmetic content correctness was **12/16 in both conditions**;
retaining the original non-arithmetic scores gave25/32 and24/32 overall.
This post-outcome diagnostic is not a confirmatory endpoint and carries no new
significance test. The frozen primary result remains19/32 versus22/32. It shows
why the observed exact-match difference cannot be treated as demonstrated
arithmetic ability, creativity or persistent-affect improvement.

## Run

With the pinned local model serving the documented API:

```sh
python -m regesh.reproductions.emotionprompt \
  --output /tmp/regesh-emotionprompt-new \
  --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
python -m unittest tests.test_emotionprompt_reproduction -v
python -m regesh.reproductions.emotionprompt_check \
  --run results/papers/emotionprompt/live \
  --output /tmp/regesh-emotionprompt-audit-new.json
python -m regesh.reproductions.emotionprompt_content \
  --run results/papers/emotionprompt/live \
  --output /tmp/regesh-emotionprompt-content-new.json
```

Use an empty output directory. The study preserves old evidence rather than
overwriting it. Broader claims need more tasks, multiple decode replications,
matched neutral text, additional models, and persistent-history interventions.

The offline audit checks archived hashes, exact prompts, per-pair seeds,
generation caps, raw/parsed response agreement, token totals, pair counts,
aggregate accuracy and exact McNemar testing. It evaluates every prediction with
the pinned unmodified upstream scorer rather than trusting the runner's scores.
It makes no network or model calls and does not execute generated answers.
