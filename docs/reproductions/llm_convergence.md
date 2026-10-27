# LLM overconvergence: scoped real-model replication on DailyDialog

Primary source checked **2026-10-25**: Blevins, *Accommodation Goes Both Ways:
Studying Linguistic Convergence Between Humans and Language Models*,
[arXiv:2605.29278](https://arxiv.org/abs/2605.29278). Its Table 2 reports, for
English WildChat, mean LIWC function-word convergence **.068** for the LLM turns
versus **.035** for the user turns (Δ = **+.033**), against human-human baselines
of .036 (DailyDialog) and .022/.020 (Ubuntu user/assistant; the .040/.038 values
sometimes quoted for Ubuntu are that table's `quant` column). Top-100 noun-lemma
convergence is **.442** for the LLM versus **.141** for users. Secondary reference:
French, D'Mello & von der Wense (2024), *A Study of Linguistic Alignment in
Dialogue Systems* (HuCLLM), whose Table 3 compares SWDA lexical alignment of true
replies (.170) and ChatGPT (.151) with shuffled-response baselines (.117 each);
we borrow its shuffled-pair baseline design.

This is a **scoped method replication on a different model, corpus role and
lexicon**. It asks whether a small open chat model, replying to the same
DailyDialog utterances that real humans replied to, converges on function-word
markers more than those humans did, and whether one instruction sentence changes
that. It does not reproduce WildChat, ChatGPT, the LIWC dictionary or noun
lemmatisation; absolute levels are not comparable, directions and contrasts are.

## Measure

The asymmetric convergence of Danescu-Niculescu-Mizil & Lee (2011), Eq. 1 of
Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg (2012):
`C^m(B→A) = P(E_u2^m | E_u1^m) − P(E_u2^m)`, estimated over exchanges (u1 from
A, u2 from B) and defined only when A exhibits m at least once. We use the
existing implementation `regesh.bonding.synchrony.CoordinationCounts` with its
eight default markers (personal pronouns, impersonal pronouns, articles,
conjunctions, prepositions, auxiliary verbs, adverbs, quantifiers) and report
"aggregated 3" (mean over defined markers). Each response set is treated as one
speaker over all 200 exchanges. The lexicon in that module is an **authored open
closed-class word list, not LIWC**, so per-category levels differ from the paper.
A nine-category mean including negation (closer to the paper's LIWC mean, where
negation was the LLM outlier at .213) is reported as a secondary statistic.
Language style matching (Ireland et al. 2011; `synchrony.lsm`) and a content-word
proxy (top-100 most frequent non-function tokens of length ≥ 4 in the selected
target utterances, scored with the same Eq. 1; a WordNet-free, POS-free
substitute for the paper's noun lemmas) are reported alongside. No module code
was changed; the runner's vectorised Eq. 1 is checked against
`CoordinationCounts` on every run and in tests.

## Data and provenance

DailyDialog (Li et al. 2017, CC BY-NC-SA 4.0), train split, 11,118 dialogues.
The original host `http://yanran.li/files/ijcnlp_dailydialog.zip` was a parked
domain on the check date (a 980-byte HTML sale page), and Wayback Machine copies
were unreachable through the session proxy. We used the ParlAI mirror
`https://dl.fbaipublicfiles.com/parlai/dailydialog/dailydialog.tar.gz`
(archive dated 2018-12-11; sha256
`c3adb09bd715b9fa5cd1ac41613b7de61eb5afbe477826a6146abefef573e6bb`; `train.json`
sha256 `8142b924bc9e382c54f172b0a8fddf5698f2c9d47479219ed11b6463aedab0b8`),
which holds the same 11,118 train dialogues in the original tokenised text. The
runner also reads the original nested-zip layout if it is present in the cache.
Raw data stays in the gitignored `.cache/dailydialog/`; only the selected short
excerpts (contexts, target utterances, human replies) and generated replies are
stored under `results/papers/llm_convergence/` for audit, under the dataset's
non-commercial share-alike terms.

DailyDialog text is tokenised (`don ' t`, `Let ' s`, space before punctuation).
The lexicon would miss every such contraction, so one declared normalisation
(straight apostrophes, contraction rejoining, punctuation spacing) is applied
identically to contexts, human replies and model replies before measurement.

## Locked protocol

Frozen in `configs/reproductions/llm_convergence.json` (`protocol_frozen_utc`,
`items_sha256`) and `results/papers/llm_convergence/protocol/freeze.json` before
any model call on the items:

* Dialogues with 4–8 turns; exact duplicates dropped (first kept).
* Cut point `k ∈ {2, 3, 4}` from `SHA256("{seed}:cut:{dialogue}")`; context is
  turns 1..k, the target `u1` is turn k, the human reply `u2` is turn k+1; both
  need ≥ 3 lexicon tokens; the dialogue is dropped if k+1 exceeds its length.
* **N = 200** items: the first eligible dialogues by ascending
  `SHA256("{seed}:select:{index}:{dialogue}")`, seed 29278. Cut points landed on
  k=2: 70, k=3: 86, k=4: 44 items. No model output was inspected.
* Two conditions. **plain**: system prompt "You are the second speaker in a
  casual two-person conversation. Reply with one short natural line." and the
  context as alternating messages, the target speaker as `user`, the replier as
  `assistant`, ending with `u1` as the final user message (for even k the replier
  opened the dialogue, so the first context message is an assistant turn).
  **bounded**: identical plus "Keep your own natural wording; do not echo or
  mirror the other speaker's phrasing.", derived from the platform's mid-range
  coordination target.
* 200 × 2 = **400 calls**, `max_tokens` 48, temperature 0, decoding seed
  29278 + item index, condition order alternating per item; every request and
  raw response is recorded; a provider error counts as an empty reply and keeps
  the row. Reply cleanup (first non-empty line, leading speaker label and quotes
  removed, then normalisation) was prespecified.
* Analysis: per-marker Eq. 1 and aggregated 3 for the human, plain and bounded
  sets; a 200-permutation shuffled-pair baseline (replies re-paired with other
  items' `u1`, seed 292781) with mean and 95% band; a paired item bootstrap
  (10,000 resamples, seed 292780, same resampled items for all three sets) for
  plain−human, bounded−human and plain−bounded; LSM means; the content-word
  proxy; mean reply length; empty/failed fraction.
* Endpoints: direction and size of the plain−human gap versus the published
  +.033 (eight-marker mean derived from Table 2: +.021); content proxy plain vs
  human versus .442 vs .141; whether the bounded sentence shrinks the gap
  without emptying replies.

Model: the pinned local Qwen2.5-Coder-1.5B-Instruct Q4_K_M served by llama.cpp
(`results/runtime/model.json`), one slot, CPU only. After the freeze and before
the first item call, two smoke calls with toy prompts (not items) confirmed that
the server accepts an assistant-first context and measured 0.16 generated
tokens/s under concurrent CPU experiments; because the run could outlast any
single tool session, a `--resume` option (continue an interrupted run with
identical config and items, no duplicate calls) was added to the runner before
the first item call and recorded in `freeze.json`. Nothing in the selection,
prompts, decoding, ceiling or analysis changed.

## Executed result

All **400** planned calls completed (started 2026-10-25 23:39 UTC, finished
01:22 UTC; 102.5 minutes of recorded model time, median call 1.6 s, calls of
30–80 s while sibling CPU experiments ran) with **zero provider errors**. Finish
reasons were `stop` for 198/200 plain and 199/200 bounded replies; three replies
hit the 48-token cap. No output had more than one line or a speaker label. One
plain and one bounded reply to "A return ticket, please." were prices
("£15.50.", "£25.00") with no alphabetic token and count as empty for the lexicon
(0.5% each). Actual prompt/completion tokens were 16,759/2,462 (plain) and
20,359/2,485 (bounded; the extra sentence adds prompt tokens). Mean reply
length in lexicon tokens: human **11.9** (median 9), plain **8.4** (median 6),
bounded **8.5** (median 7). LSM(u1, reply) over non-empty replies: human .545,
plain .559, bounded .565.

**Primary endpoint — function-word convergence toward u1 (eight markers,
aggregated 3, 200 exchanges per set; paired item bootstrap 95% CI; shuffled-pair
baseline mean and 95% band over 200 permutations):**

| Response set | Convergence | Bootstrap 95% CI | Shuffled baseline [95% band] | Nine categories incl. negate | Content-word proxy | Shuffled proxy |
|---|---:|---:|---:|---:|---:|---:|
| human (real next turn) | **+.034** | [+.007, +.061] | +.001 [−.021, +.026] | +.041 | .062 | −.000 [−.008, +.010] |
| plain model | **+.025** | [+.002, +.050] | +.001 [−.021, +.021] | +.026 | .081 | +.001 [−.009, +.012] |
| bounded model | **+.032** | [+.009, +.056] | +.000 [−.018, +.020] | +.030 | .081 | +.001 [−.009, +.013] |

| Contrast | Function words | 95% CI | Content proxy | 95% CI |
|---|---:|---:|---:|---:|
| plain − human | **−.009** | [−.041, +.025] | +.019 | [−.017, +.063] |
| bounded − human | −.002 | [−.034, +.031] | +.019 | [−.012, +.058] |
| plain − bounded | −.007 | [−.023, +.009] | −.000 | [−.022, +.021] |

All three true values lie above the upper limit of their own shuffled band, so
each response set converges to its targets beyond chance pairing, as humans and
LLMs both did in the paper. `accommodation_assessment` places all three sets
`within` the published human-human band.

| Marker | human | plain | bounded | Blevins DailyDialog | Blevins WildChat user | Blevins WildChat LLM |
|---|---:|---:|---:|---:|---:|---:|
| ppron | +.017 | +.023 | +.013 | +.028 | +.048 | +.086 |
| ipron | +.029 | +.042 | +.044 | +.048 | +.031 | +.051 |
| article | +.131 | +.013 | +.037 | +.046 | +.028 | +.041 |
| conj | −.020 | +.008 | +.015 | +.038 | +.021 | +.035 |
| prep | +.012 | +.043 | +.057 | +.030 | +.018 | +.022 |
| auxverb | +.021 | +.029 | +.020 | +.015 | +.022 | +.033 |
| adverb | +.025 | +.045 | +.055 | +.039 | +.032 | +.063 |
| quant | +.056 | +.002 | +.017 | +.084 | +.033 | +.067 |
| negate (secondary) | +.094 | +.035 | +.015 | −.005 | +.081 | +.213 |

The published columns are LIWC-based and not level-comparable; they are shown
for direction only.

### Comparison with the published finding

* **Direction of the LLM−human gap: not reproduced.** Blevins reports the LLM
  converging about twice as much as its users (+.033 on the LIWC mean; +.021 on
  the eight markers used here, derived from Table 2). Here the plain model
  converged slightly *less* than the real human repliers, **−.009 [−.041,
  +.025]**. The interval includes zero and barely includes the derived
  published eight-marker gap (+.021), so this study cannot distinguish "no
  overconvergence" from a gap of the published size on this lexicon; it does
  rule out a gap of +.025 or more on this measure. Nothing here resembles the
  twofold excess, and negation, the paper's outlier (.213 vs .081), was the
  marker on which the humans converged most and the model least.
* **Content-word proxy: same direction, far smaller.** The model's proxy
  convergence exceeded the humans' (.081 vs .062, +.019 [−.017, +.063]), the
  same sign as the published .442 vs .141, but at roughly a tenth of the
  published absolute level and with an interval including zero. The proxy is
  not noun-lemma convergence and the vocabulary is drawn from 200 short
  utterances, so even the sign is weak evidence.
* **Human-human level.** The real DailyDialog repliers scored +.034 (eight
  markers) and +.041 (nine), close to the paper's DailyDialog .036/.041 on
  LIWC. The per-marker pattern differs (articles +.131 vs +.046; conjunctions
  −.020 vs +.038), so the agreement of the means should not be over-read.
* **Bounded instruction.** The sentence did not reduce convergence: bounded −
  plain was **+.007 [−.009, +.023]** on function words and −.000 on the content
  proxy. It did not empty replies (0.5% empty in both conditions, mean length
  8.4 → 8.5 tokens, one fewer truncation). There was no measured gap for it to
  close, so this is a null on both sides of the endpoint.

The model's replies are 30% shorter than the human ones. Eq. 1 compares a
conditional and a marginal probability for the same speaker, and the shuffled
baselines near zero show that the length difference creates no spurious
convergence, but shorter replies exhibit markers less often and so cap the
reachable coordination; a length-matched comparison was not run.

### Analysis correction after the run

The first `summary.json` written by the runner applied each shuffled-baseline
permutation to targets and replies together, which preserves every pairing and
made the "shuffled" mean equal the true value with a zero-width band. The
shuffle was corrected to permute replies only, as prespecified, and the summary
was recomputed from the recorded rows with
`python -m regesh.reproductions.llm_convergence analyze`. No call, row, point
estimate or bootstrap interval changed; `manifest.json` records the superseded
summary hash, the corrected module hash and the reason, and the `runner.py`
copy in `live/` remains the version that made the calls. The two-condition
contrasts and the comparison above were unaffected.

The protocol, config, locked items, every raw request/response, the per-call
rows and the summary are in `results/papers/llm_convergence/`
(`protocol/items.jsonl` sha256
`56f10e219d98259827469624eed7b5d265022ede788ce1ae861147997a95db47`).

## Run

```sh
python -m regesh.reproductions.llm_convergence prepare \
  --cache .cache/dailydialog --config configs/reproductions/llm_convergence.json \
  --items results/papers/llm_convergence/protocol/items.jsonl
python -m regesh.reproductions.llm_convergence run \
  --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder \
  --config configs/reproductions/llm_convergence.json \
  --items results/papers/llm_convergence/protocol/items.jsonl \
  --output /tmp/regesh-llm-convergence-new
python -m regesh.reproductions.llm_convergence analyze \
  --run results/papers/llm_convergence/live --output /tmp/llm-convergence-summary.json
python -m unittest tests.test_llm_convergence_reproduction -v
```

`prepare` needs the archive in the cache directory (original zip or the ParlAI
`parlai/dailydialog.tar.gz` extracted beside it). `run` requires an empty output
directory; `analyze` recomputes the summary from recorded rows without any model
call. The tests use an inline dialogue fixture and a fixture model; no network.

## Limitations

* **Model.** A 1.5B-parameter quantised *coding* model on CPU, not ChatGPT
  (GPT-3.5/4). One greedy decode per item; decoding variance is unmeasured.
* **Lexicon.** An authored open function-word list, not LIWC 2007. Category
  membership and coverage differ, so absolute convergence levels are not
  comparable with Table 2; only directions and within-study contrasts are.
* **Corpus role.** Human replies are the real next DailyDialog turn after a
  2–4 turn scripted everyday dialogue; the model answers the same `u1` in the
  same position. WildChat is organic multi-turn task-oriented chat, where user
  turns are far longer and the LLM's turns are long answers.
* **Noun proxy.** Frequent non-function tokens of the target utterances without
  POS tagging, lemmatisation or WordNet filtering, so it is not noun-lemma
  convergence; it also uses 200 target utterances rather than a corpus.
* **Quality.** Only reply length, finish reason, empty-reply and provider-error
  counts describe output quality; coherence and appropriateness are unmeasured,
  and a reply that goes off-topic or ignores the context is still scored.
* **Aggregation.** Each response set is one aggregated speaker; there is no
  per-speaker macro average and no per-turn trajectory as in the paper.
* **Data.** Mirror rather than the original archive; identical dialogue count
  and text format, but the original zip's checksum could not be verified.
* **Timing.** Elapsed times were recorded under heavy concurrent CPU load and
  are descriptive only.
