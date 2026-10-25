# Executed procedural creativity and transfer experiment

This experiment tests a specific causal hypothesis: **learning consequences of
search actions and regulating exploration versus verification using persistent
state improves constrained solution search.** It is CPU numerical research,
not a live-LLM evaluation, a human study, or evidence of subjective experience.
The full study is exploratory: review led to structural corrections after an
initial pilot. See [pilot provenance](exploratory_pilot.md). No held-out
performance-driven coefficient optimization was performed.

## Reproduction and evidence

Run `python -m regesh.benchmark --output results/benchmark`. Configuration is
`configs/benchmark.json`. The runner saves exact config, generic trained weights,
source snapshots and hashes, Python/NumPy versions, initial per-trial snapshots,
training transitions, development results, test results, action-by-action traces,
model forecasts, and paired seed-cluster bootstrap intervals. Wall timings vary;
seeds, generated designs, training weights, actions, and scores are deterministic
on a fixed numerical runtime. `python -m unittest discover -s tests -p
'test_benchmark.py' -v` checks budgets, split leakage, label-clone identity,
fixed trained weights, source model serialization, and constraint visibility.

## Tasks and executable scoring

An idea is a canonical six-dimensional design with four levels per dimension.
Knob meanings differ across software caching/replication/batching, engineering
storage/insulation/demand-shifting, and scientific-design sample size/blocking/
controls. `SEMANTICS` and `DesignTask.evaluate` define every mechanism, resource,
robustness, and utility equation. All are transparent toy equations, not validated
physical/software/scientific models.

A trial has 32 objective probes. Available actions are random divergence, one-knob
local refinement, combination with retrieved evidence, and verification of the
incumbent on another workload. At probe 17 a resource constraint tightens by 15%;
that change is visible before selection. Old observations remain in trace history;
obsolete estimates are invalidated. Improvement after criticism is quality change
from the incumbent at revision to the final solution, both scored under the
revised constraint. This is accurate external constraint feedback, not a language
critic or social approval manipulation.

Final adjudication is blind to condition: 64 independent fixed workloads score
objective usefulness, feasibility, and utility discounted to 25% when a constraint
fails. Each trial has 64 pre-revision and 64 final adjudication checks, outside
the policy's accessible probe budget, equal for all conditions. No counterfactual
rollout makes extra simulator calls. Learned predictions use only arithmetic.

Originality means a final canonical design is absent from the four-design
`REFERENCE` corpus. This is a weak coverage test (4 of 4096 designs), not published
novelty. `mechanism_original` separately compares the subset of enabled knobs to
reference subsets. `mechanism_diversity` counts distinct enabled-mechanism subsets
(out of 64), and `canonical_design_diversity` counts distinct knob configurations
(out of 4096). Neither unusual language nor semantic distance enters any score.
Useful-original requires quality >= .45 and compliance >= .9. This threshold is
fixed and disclosed; stronger reference coverage is a necessary follow-up.
Factual support is limited to executable equations; linguistic factual support,
human usefulness, and calibration of verbal claims are not measured.

## Learning, development, and held-out trials

Generic training uses random-policy transitions on software and engineering
families: eight seeds, 12 episodes per seed/family, 32 probes each (6144 probes).
Development has four distinct seeds on these two families; test uses 20 distinct
seeds across all three families (660 trials for 11 conditions). Scientific design
is never used for training. This is a modest formula/semantic transfer test:
all families share a six-knob grammar and action operators. There are no simulated
people, independently held-out social partners, or human longitudinal data.

Per-trial state has signed prediction error/mood, resource burden, uncertainty,
progress, action-specific expectations, observations, and candidates. Actual
outcomes update expectations and appraisal within that history. Every trial starts
from the complete declared initial snapshot; no cross-trial adaptation occurs.
Trained generic weights remain fixed. Agent-specific development and generic
model fitting are distinct. Context includes 17 measurements, all available to
every policy, including burden and complete numerical regulation summaries.

The hybrid trains a ridge action-conditioned transition predictor with four
heads: incumbent quality gain, uncertainty reduction, burden change, and constraint
compliance. Its 18 local features (bias plus 17 inputs) in four action blocks and
four output heads give 288 parameters. Its score is

`gain + (.03 + .08*uncertainty)*uncertainty_reduction - (.02 + .08*burden)*burden_change + .04*compliance + .015*negative_signed_error*(action==diverge)`.

Coefficients were fixed as engineering choices, not learned discoveries. The
internal burden has independent action-dependent decay/recovery and affects the
controller, but does not disable actions or constrain the external simulator.
Benefits could therefore arise from regularization or resource-sensitive search;
this is not evidence that burden itself is biologically meaningful. High effort
is not equated with distress.

## Comparators and interventions

All conditions share observations, evidence-focused incumbent selection, reference
motifs, operators, and budgets. Every condition executes four predictor calls per
decision, including ignored calls for fixed baselines, plus one diagnostic
feasibility forecast. No LLM, token, external retrieval, or tool calls occur.
Model invocation counts and measured runtimes are recorded. Numerical feature
construction has slightly different overhead across model classes.

1. `memory_planning`: cyclic divergence/refinement/combination/verification.
2. `static_persona`: fixed optimistic/exploratory action schedule.
3. `turn_sentiment`: only the immediately preceding quality selects strategy.
4. `heuristic`: persistent burden, stagnation, and mood regulate strategy.
5. `hybrid`: learned counterfactual transition prediction with persistent control.
6. `ordinary_adaptive`: a structurally different direct contextual action-value
   model, with the identical 17 inputs and the same observed scalar objective.
   It uses 72 nonlinear local terms/action (raw values, squares, cubes, adjacent
   products, three cross-products); one head has 288 parameters, all active in
   selection. This matches parameter count, information, actions, and arithmetic
   budget, while effective inductive biases differ. It is a strong ordinary
   adaptive comparator, not an otherwise identical affect module renamed.
7. `unlabelled_hybrid`: exact computation of the hybrid with labels removed;
   identical actions are expected and demonstrate no benefit from labels.
8. `frozen`: regulation and adaptation summaries frozen, observation memory intact.
9. `shuffled`: regulation summaries are replaced with a sampled earlier summary
   within the same history. This destroys temporal alignment; it is not an exact
   uniform permutation preserving final marginals.
10. `no_memory`: removes the mood-times-prediction-error term in candidate-donor
    retrieval ranking, retaining observations, evidence access, and all other
    adaptation. This disconnects one affect-memory coupling, not all memory.
11. `no_counterfactual`: same learned model, same input and predictions, selection
    uses the predicted gain head alone. This removes projected internal-state
    contributions, rather than replacing the learned policy by a heuristic.

Static-persona and sentiment conditions are numerical analogues. They do not
satisfy a live-LLM comparison between prompt personae and emotional state. The
shared outcome predictor's Brier score is evaluated on the actions each policy
selects; it is a diagnostic of this common predictor, not policy-specific
confidence calibration.

## Analysis and limitations

Primary endpoint is blind held-out executable quality. Tables separately report
usefulness, compliance, corpus-relative novelty, diversity, post-constraint
improvement, computation, and timing. Paired intervals resample whole seeds,
keeping the three family outcomes clustered. Multiple comparisons are exploratory
and not multiplicity-corrected. There is no claim that a favorable interval alone
establishes a new affect mechanism or real-model creativity. A negative result
against ordinary adaptation must be reported even if the hybrid beats fixed
baselines. No human judgments, physiological measures, longitudinal participants,
or independently reviewed design evaluations were collected.

The most informative next experiment is the same fixed-budget strategy competition
with a real base model on executable software-design tasks, stronger prior-art
retrieval, independently held-out problem families, and a direct adaptive comparator.
Count every appraisal, reflection, retrieval, and tool invocation in the shared
budget; blind final evaluators; preserve all prompts/responses and complete initial
agent snapshots. Keep generic weights frozen and allow only declared within-trial
adaptation. The current numerical runner provides no substitute for this study.
