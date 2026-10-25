# Independent review of paper reproduction claims

Review opened **2026-10-25**, before the new paper experiments were inspected.
This reviewer does not implement or select the three mechanisms. The review
checks what the executions establish and which comparisons are justified.

## Terms and acceptance criteria

The [National Academies' 2019 report](https://www.nationalacademies.org/read/25303/chapter/3)
distinguishes computational reproduction with the original inputs and procedures
from replication using newly obtained observations. The terms below make the
actual fidelity of this project more explicit:

| Label | Required evidence | What it does not establish |
|---|---|---|
| Exact computational reproduction | Author code or independently verified identical computation, original input data and configuration, a named published output, and a numerical agreement check | Replication with new data or robustness to another implementation |
| Independent implementation of a published experiment | A named experiment, source-supported equations/environment/parameters, newly implemented execution, and direct quantitative comparison | Identity with missing author code or undisclosed details |
| Scoped method replication | The published mechanism and dependent variable, explicit model/task/scale substitutions, fixed execution protocol and new raw observations | Reproduction of the original absolute score or all claims in the paper |
| Conceptual demonstration | A new environment inspired by the paper, or only checking a formula on fixtures | A reproduced published experiment |

For the user's request, each of at least **three distinct papers** needs an
executed experiment rather than another literature entry. Each report should
name its figure/table/section, publication status, primary source, available code,
input/configuration hashes, executed command, raw output, comparison endpoint and
fidelity label. A conceptual demonstration must not be silently counted as a
paper reproduction. If no suitable original numerical table exists, compare a
source-supported analytic prediction or published direction and explicitly state
that original figure values were not recovered. A qualitative plot resemblance
does not establish matching published numbers.

The [AAAI reproducibility checklist](https://aaai.org/aaai-24-conference/reproducibility-checklist/)
also supports recording randomization, hyperparameters and uncertainty. A seed
alone is insufficient: specify the generator, draw order, runtime versions and
which observations share a seed. Fresh live model responses need not be bitwise
repeatable even when request seeds match.

## Main failure modes to guard against

1. **Different model or task called an exact reproduction.** Running EmoBench
   with a small local coding model tests its benchmark performance; it does not
   reproduce a GPT-4 result. Likewise, a new multi-armed bandit with an IAC-inspired
   score is useful only as a conceptual demonstration unless it matches an
   actual paper experiment.
2. **Comparing incompatible denominators.** Whole-item QA accuracy, average
   reward, set-level F1 and a deterministic reward curve are different constructs.
   Do not average or rank these as a common affect-quality score. Compare each
   current endpoint with its corresponding published endpoint and report units.
3. **Pseudo-replication.** Trials, seeds, task families, question variants and
   time steps are not interchangeable independent sample units. Use paired
   seed-level or source-scenario-level uncertainty where appropriate. The number
   of time points does not enlarge the number of independent agents.
4. **Missing original uncertainty.** A published mean without raw runs, variance
   or independent sample count permits descriptive discrepancy, not a formal
   conclusion that the current method is statistically better or equivalent.
   Failure of a difference test does not establish successful equivalence.
5. **Post-hoc choices hidden as prospective decisions.** Preserve the selected
   subset and protocol before inference. Separate implementation corrections,
   runtime retries and parser changes; do not discard failures or choose whichever
   metric agrees with the source. Do not tune thresholds on the same outcomes
   that are reported as replication tests.
6. **Data reuse mistaken for new evidence.** Recomputing author predictions is
   an exact analysis check. It adds no new independent model observations. A
   bounded new-model run should have a separate denominator and result table.
7. **Theory validated with its own equation.** Checking an analytic reward
   function's monotonicity confirms implementation and the calculation. It is
   not external validation of animal behavior, affect, learning or creative
   problem solving. A learned-policy experiment is a separate endpoint.
8. **Affective interpretation overextended.** Benchmark emotion knowledge,
   homeostatic regulation and learning progress are relevant antecedents. None
   alone establishes persistent subjective emotion, psychologically valid traits,
   or improved useful idea generation in the Regesh controller.

## Numerical comparison template

For each paper retain: **published value/claim; current value; independent unit
count; uncertainty; implementation/model/task deviations; agreement criterion;
result and scope**. A missing published value stays missing. If reading points
from a figure, identify digitization, axes, error and tool; do not invent decimals.
Report absolute percentage-point changes separately from relative changes. When
the local run uses a different model, call the difference a descriptive benchmark
gap and state that model, quantization and protocol differences are confounded.

## Review status

The criteria above are a prospective review framework, not a claim that all new
experiments have passed. The checks below identify the artifacts actually read.

## Independent source-data check: Croissant et al. (2024)

The reviewer independently read the three released CSV files with Python's
`csv` module, independently calculated repeated-measures sums of squares, and
used SciPy 1.17.0 `f_oneway`, `f.sf` and `ttest_rel` to check results. This was a
separate short calculation, not a call to the experiment implementer's scorer.
The exact source files are identified in the experiment's download manifest.

| Endpoint | Reviewer recomputation from full released data |
|---|---:|
| STEU correct, no memory / memory / appraisal | 24 / 31 / 35 out of 42 |
| Sentence Authentic means, conditions 0 / 1 / 2 | 61.511818 / 61.886250 / 82.604444 |
| Sentence Authentic valid observations | 22 / 24 / 27, total 73 |
| Sentence Authentic independent-group ANOVA | F(2,70)=3.3017389791, p=0.0426318640 |
| User-study naturalness, complete-subject repeated measures | F(2,58)=7.7041677320, p=0.0010784398 |
| User-study input response, complete-subject repeated measures | F(2,58)=6.0825439783, p=0.0039985329 |
| User-study emotional sensitivity, complete-subject repeated measures | F(2,58)=5.5187362234, p=0.0063966150 |

The user-study file has **90 unique participant-condition cells, 30 participants,
three conditions each**. Its full-data naturalness means are 4.466667, 4.966667
and 5.700000. The printed article's corresponding means and F statistics do not
all match these calculations. The article describes a within-subject analysis
but reports denominator degrees of freedom of 84 for several items. A conventional
complete-subject three-condition analysis here has denominator df=58. Unknown
exclusions, coding or analysis choices must remain unknown; do not search for a
participant subset that happens to reproduce printed numbers.

The sentence means match the paper after rounding. Its printed Authentic
F=5.10, p=0.03 does not match the conventional independent-group ANOVA above,
but numeric regression on condition codes 0/1/2 gives **F(1,71)=5.0933248098,
p=0.0270971577**, recovering that reported pattern. This numeric trend tests a
different estimand from a categorical condition effect. The code/paper's displayed
precision does not support claiming bit-for-bit identity of all printed values.
Sentences from a small fixed set of generated conversations are also not
independent replications of an agent architecture. A reanalysis of the released
LIWC scores avoids substituting an unlicensed dictionary, but it does not rerun
the proprietary LIWC extraction or regenerate the conversations.

These are **computational reanalyses of original data**, with both agreements
and unresolved numerical discrepancies. They are not new participant results,
new GPT-3.5 observations, or confirmation that appraisal specifically improves
useful problem solving. The conventional repeated-measures sensitivity analysis
is an additional analysis; it must not be represented as the original analysis.

Across the 12 user-study questionnaire items, the independent Holm adjustment
gives naturalness p=0.0129413, input response p=0.0439839, and emotional
sensitivity p=0.0639661. Only the first two survive a 0.05 family threshold.
These assume sphericity. Independently computing Greenhouse-Geisser epsilon
from the eigenvalues of within-participant-centered covariance gives respective
epsilons **0.9937513, 0.7302040 and 0.8553219**. With Greenhouse-Geisser and Holm
across all 12 items, the corresponding p values are **0.0132953, 0.1040053 and
0.1040053**: only naturalness survives. The correction, family and unadjusted
values must remain visible; selecting among them to maximize significance is
not justified. All independent calculations and source hashes are retained in
[reviewer checks](../../results/papers/reviewer_checks.json).

The main agent also added an explicitly post-hoc ordinal sensitivity on all
12 items. The reviewer independently implemented within-participant average
ranks, the tie correction, the Friedman rank-sum statistic, and Holm adjustment
without calling that runner or SciPy's Friedman routine. All 12 statistics agree
exactly with the runner. Friedman plus Holm retains naturalness, responsiveness
and emotional sensitivity; GG-ANOVA plus Holm retains naturalness only. These
are model-dependent analyses of the same observations, not three independent
confirmations. The Friedman p values use an asymptotic chi-square approximation
with three conditions. Naturalness is the common result under these two models;
the additional findings are sensitive to the inferential choice.

## Pre-result implementation checks

The reviewer read the three new executable modules and ran their **18 focused
tests**, all passing: seven homeostatic, four happiness-bandit and seven
EmotionPrompt tests. These tests check equations, boundaries, scoring and runner
mechanics; they do not count as scientific outcomes.

- **Keramati & Gutkin (2014), Figure 6:** the one-dimensional drive is
  `abs(H) ** (4/3)` and the update uses drive reduction plus discounted successor
  value. Every simulated agent has its own Q table. The allocation guard raises
  instead of reflecting/clipping trajectories. The original source does not
  resolve the softmax beta convention in the bounded check; the declared primary
  inverse convention and opposite-convention sensitivity must remain separate.
  The 32 long-run replicas extend a paper panel based on one realization. Their
  pooled visit histogram is not a histogram of the original single agent.
- **Dubey et al. (2022), Experiments 2a/2b:** the learning rules, UCB equation,
  selected exploration/learning parameters, aspiration schedules, bandit means
  and random walks match the inspected primary PDF. Controller comparisons vary
  epsilon and sometimes learning rate alongside reward definition; they do not
  isolate an affective reward effect. The additional fixed-aspiration versus
  optimistic ordinary-learner check holds epsilon, learning rate and randomness
  constant. The Q-value translation establishes the intended equivalence for
  both sample-average and constant-step updates; action agreement still needs
  empirical checking because floating-point paths can diverge.
- **Li et al. (2024), EmotionPrompt:** the intervention changes only the declared
  suffix. The scoring rule preserves the pinned benchmark's unusual first-line,
  period and punctuation normalization, with a separate strict score. Original
  execute items have external gold answers. The chosen four-task/32-item sample,
  small quantized coding model, reconstructed delimiter and single suffix differ
  from the published aggregate; no precise original-effect reproduction follows.
  The missing length-matched neutral control also prevents attributing a local
  difference specifically to emotional semantics.

## Final review of the four executed studies

All four studies now have actual outputs, source/configuration snapshots,
quantitative comparison tables and runnable commands. This meets the requested
minimum of three papers **with explicitly different reproduction scopes**; it
does not amount to four exact original-code reruns. The reviewer ran the updated
four focused suites: **29 tests passed**, including nine Chain-of-Emotion checks
and the added arithmetic-content check. Independent numerical checks below do
not import the mechanism implementations.

| Paper and scope | Quantitative finding checked | Review conclusion |
|---|---|---|
| Keramati & Gutkin (2014), full-scale independent Figure 6 implementation | Directly reading all 32 primary NPZ batches gives 100,000 terminal states, mean 0.04668, mean absolute deviation 5.41912, SD 6.93353, history maximum 86; 86 agents exceed the paper's maximum 61 | Regulatory learning works versus the declared random walk, but the reported tail extreme is **not reproduced**. The unresolved beta convention and missing author code prevent diagnosing why. Do not attribute the difference to seed alone. |
| Dubey et al. (2022), independent Experiments 2a/2b implementation | Stdlib recomputation gives stationary-spread objective 4396.97 (SD 238.21), fixed 4301.38 (SD 1019.73); sudden-change objective 4195.78 (SD 190.15), fixed 4566.60 (SD 234.13). Every one of 8,000 fixed/optimistic paired round rewards is identical | Sudden-change benefit closely agrees with the paper; the stationary-spread fixed-comparison benefit reverses. Both agreement and failure must be reported. The ordinary optimistic learner is a substantive causal control, not an affect-specific success. |
| Croissant et al. (2024), original-data computational reanalysis | All three STEU counts exactly match. Released LIWC means and numeric-trend F pattern are recovered; full released participant statistics do not all match printed values | Exact count reproduction plus partial statistical correspondence and identified discrepancies. Extra paired/GG/ordinal analyses are sensitivity checks on original data, not new participants or new agent trials. |
| Li et al. (2024), scoped new-model EmotionPrompt replication | Pinned upstream scorer on all 64 records gives baseline 19/32, EP02 22/32; five gains, two losses; exact paired p=0.453125. Prompt/completion token totals are 1624 versus 1819 | A small inconclusive difference, with different model/subset and unequal input tokens. It does not reproduce the original aggregate or establish an emotion-specific improvement. |

The reviewer checked all EmotionPrompt raw-call/row identities, single-message
requests, temperature zero, 32-token ceilings and original upstream exact-match
scores. There were no additional model calls. Its post-hoc arithmetic diagnostic
accepts only complete integers or correctly specified original-operand equations;
the original primary metric remains intact. Correct arithmetic content is 12/16
in **both** conditions, while the combined diagnostic is 25/32 versus 24/32.
Thus even the direction depends on format sensitivity. That diagnostic is useful
for locating the confound, not for replacing the frozen endpoint or running a
new favorable significance test.

The primary papers establish relevant antecedents across homeostatic regulation,
comparative reward, appraisal prompting and static emotional wording. These
executions neither establish that the Regesh hybrid improves useful original
solutions nor erase its earlier negative controller comparison. No common score
combines the four incomparable dependent variables. A reproducibility failure is
reported as an unresolved result under the declared implementation rather than
a claim of misconduct or a refutation of an entire theory.

The highest-value follow-up depends on the claim: obtain authors' original
bandit/Figure 6 code and exact policy conventions to diagnose the numerical
failures; for an affect-specific behavioral claim, use a prospectively fixed
real-model experiment with ordinary adaptive and length-matched neutral controls,
adequate baseline task success, and scores that separate correct content from
format compliance. More correlated metrics alone will not resolve causality.
