# Chain-of-Emotion: reproducing the published analyses

Search and execution date: 2026-10-25. Primary source: Croissant, Frister,
Schofield and McCall (2024), [An appraisal-based chain-of-emotion architecture
for affective language model game agents](https://doi.org/10.1371/journal.pone.0301033),
PLOS ONE 19(5), e0301033, peer reviewed. The article is CC-BY. The
[authors' OSF project](https://doi.org/10.17605/OSF.IO/QPT6Z) has three public CSVs;
its API reported `node_license: null`. Their contents are kept in an external
cache, and the runner verifies the OSF-supplied SHA-256 hashes. Public access
is not represented as a blanket data redistribution license.

This is a **statistical reproduction on the authors' original data**. It
re-executes scoring and statistical analyses for three experiments within one
paper. Those three experiments count as one reproduced paper. It makes zero
new model calls and recruits zero new participants. It does not recreate
the original GPT-3.5 outputs, licensed LIWC dictionaries or user experience.

```bash
python -m pip install -e '.[papers]'
python -m regesh.reproductions.chain_emotion \
  --cache .cache/papers/chain_emotion --fetch \
  --output results-new/chain-emotion --seed 20261025 --bootstrap-samples 10000
```

Outputs must be absent or empty. Omitting `--fetch` runs offline from the
verified cache. The source data are 16,445 bytes (STEU), 56,872 bytes (LIWC),
and 4,827 bytes (user ratings). The saved manifest records URLs, hashes,
software versions, seed, analysis duration and runner hash. The protocol is
also recorded in `configs/reproductions/chain_emotion.json`.

The implemented analysis is exploratory: initial inspection of the public data
revealed discrepancies before the final analysis was fixed. It is not described
as a preregistered confirmation. No exclusions were searched or chosen to force
agreement with published numbers.

## Experiment 1: emotional-understanding scores

The original paper used 42 Situational Test of Emotional Understanding (STEU)
items, answered by GPT-3.5 under No Memory, Memory and Appraisal Prompts.
The released CSV contains the answer choices and author-coded correctness;
it does not constitute an independently reviewed answer key. We sum those
score columns without re-labeling the answers. Copyrighted test items or a new
copy of the full STEU are not redistributed.

| Condition | Published and reproduced | Accuracy | Wilson 95% interval |
|---|---:|---:|---:|
| No Memory | 24/42 | 57.14% | 42.21–70.88% |
| Memory | 31/42 | 73.81% | 58.93–84.70% |
| Appraisal Prompts | 35/42 | 83.33% | 69.40–91.68% |

The same 42 items are paired units. Compared with No Memory, Appraisal Prompts
gains 12 correct items and loses one: +26.19 percentage points, paired-item
bootstrap 95% interval [11.90, 40.48], exact McNemar p=.003418 and Holm p=.010254
across three pairwise contrasts. Compared with Memory, the incremental gain is
four items (+9.52 percentage points), with exact McNemar p=.21875. The large
appraisal-versus-no-memory difference therefore does not isolate an effect
beyond the memory/context baseline in this one fixed run.

Item-level resampling describes variation over these tasks, not over model
runs or human populations. The paper supplies one recorded answer per item and
condition, so inference-seed variation cannot be estimated.

## Experiment 2: LIWC analysis and condition coding

The released file contains 75 sentence segments from three fixed generated
conversations. Two Chain-of-Emotion segments contain punctuation only and lack
derived LIWC metrics. The common complete-case set retains 73 segments
(No Memory 22, Memory 24, Chain of Emotion 27) for all six reported metrics.

The original authors' computed scores are reanalysed; applying LIWC itself
is not reproduced. All six reported F/p patterns are closely recovered by a
linear trend treating condition codes 0/1/2 as a numeric regressor. This is
distinct from treating the three architectures as an unordered categorical
factor:

| LIWC authenticity analysis | F | df | p |
|---|---:|---|---:|
| Published | 5.10 | 1,71 | .03 |
| Numeric 0/1/2 trend | 5.09332 | 1,71 | .027097 |
| Categorical one-way ANOVA | 3.30174 | 2,70 | .042632 |

The reproduced trend F rounds to 5.09, not exactly the printed 5.10; agreement
is within .01 and should not be described as an exact numerical match.

The authenticity means reproduce as 61.5118, 61.8863 and 82.6044. Small
rounding discrepancies remain; the reported Chain-of-Emotion affective-word
mean 10.65 does not match the released-score mean 10.8311. Numeric coding is
reported as an analysis diagnostic, not adopted because it yields a smaller p.
Neither the trend nor categorical tests survive Holm correction across six
outcomes. Sentences from one conversation per architecture are not independent
replications; these tests should not establish a population-level architecture
advantage.

## Experiment 3: participant-matched rating analysis

All 90 rows from 30 participants and three conditions have complete item
ratings. One participant lacks demographic age; that participant remains in
the analysis because all task outcomes are observed. We use the participant
IDs to preserve the within-subject design described by the paper.

The full released file does **not** exactly reproduce Table 3's means or
reported F values. The raw item values span 1–7; the article's caption says
0–6. The reported residual df of 84 differs from both the full-data independent
ANOVA's 87 and the repeated-measures ANOVA's 58. The source analysis code or
exclusion rules were not provided in the node. These discrepancies are left
visible, rather than guessing which records should disappear.

| Outcome | Published F, p | Paired RM F(2,58), p | Holm p over 12 items | GG + Holm p |
|---|---|---|---:|---:|
| Natural reactions | 3.65, .03 | 7.70417, .001078 | .01294 | .01330 |
| Reacted to input | 3.62, .04 | 6.08254, .003999 | .04398 | .10401 |
| Sensitive to others' emotions | 3.31, .04 | 5.51874, .006397 | .06397 | .10401 |

The uncorrected repeated-measures analysis gives two significant omnibus
outcomes after Holm correction. With Greenhouse–Geisser sphericity sensitivity
and the same Holm family, only natural reactions remains significant. An
omnibus condition test is not itself a specific Chain-of-Emotion-versus-Memory
contrast. All 36 paired contrasts, paired effect sizes and participant-bootstrap
intervals are included in `summary.json`, with a separate Holm correction.
These questionnaire results measure perceived behavior, not persistent internal
affect, biological fidelity or subjective experience.

## Relevance to regesh

This is a close architectural antecedent: appraisals are stored alongside
conversation memories and condition future responses. The reproduced results
support a narrow claim about the authors' recorded emotional-understanding
outputs, and suggest that some perceived naturalness differences remain under
paired analysis. They do not demonstrate improved useful software ideas,
development across restarts, or an advantage over a capacity- and budget-matched
ordinary adaptive controller. Appraisal adds calls and context in the original
architecture; the paper does not control those quantities as the platform's
later software experiment does.

The independent critic separately recomputed the score counts, LIWC categorical
ANOVA and all 12 repeated-measures tests from the pinned CSVs. Tests verify
source-corruption rejection, output preservation, Holm correction, paired
bootstrap and repeated-measures invariance to participant intercepts and
condition permutations.
