# Paper reproduction checkpoint

Search and execution date: 2026-10-25. User requested experiments from at least
three relevant papers, with quantitative comparisons. Existing platform studies
and historical provenance are preserved unchanged.

## Selected work, before new outcomes

- Keramati & Gutkin (eLife 2014): independent implementation of Figure 6,
  homeostatic Q-learning. Inspect the original supplementary parameter table;
  explicitly test ambiguous softmax conventions rather than tune to the figure.
- Dubey, Griffiths & Dayan (PLOS Computational Biology 2022): independently
  implement published 10-arm bandit experiments at the original 2,000 rounds and
  5,000 steps, with published fixed parameters and no outcome-based retuning.
  Test whether a fixed comparison reward is equivalent to ordinary optimistic
  initialization before assigning a uniquely affective explanation.
- Croissant et al. (PLOS ONE 2024): seek released Chain-of-Emotion data and
  computationally reproduce the reported statistics. This is a reanalysis of
  original observations, not a new participant study or new GPT-3.5 execution.
- Li et al. (ICML 2024): bounded EmotionPrompt method replication with original
  Instruction Induction tasks, fixed item selection, and the existing CPU Qwen
  model. A different model/subset cannot reproduce the original aggregate score.

An independent reviewer checks fidelity, denominators, uncertainty and causal
claims. Agents own separate modules, tests, configs, paper reports and outputs;
the main agent integrates the comparison and verifies the measurements.

## Resources

Runtime status: connected, unrestricted enforced HTTP policy, no configured
credentials/endpoints, no GPU. Four CPU equivalents and about 17 GiB RAM.
The existing pinned Qwen2.5-Coder-1.5B-Instruct Q4_K_M and llama.cpp b11878 CPU
server were restarted on loopback port 8080, alias `regesh-qwen-coder`.
`results/papers/runtime/server.log` is new; historical runtime/model metadata are
unchanged. New live ceiling: 64 calls, 32 completion tokens each (input tokens
additional); no paid service. Numerical studies use NumPy and explicit seeds.

## Completed checkpoint

All four studies and independent audits are complete. No model/simulation jobs
remain active. The local model server was stopped after all 64 paper-study
responses; its cache remains available. Historical results and trained weights
are unchanged. The combined report is `docs/paper_reproductions.md`.

- Homeostatic Figure 6: 482 million transitions. Final mean absolute deviation
  5.41912 versus no-learning39.79792. Population maximum86 versus published61
  does not reproduce;86/100000 exceed61. Long-run replicas:21/32 remain≤40.
  Opposite-softmax sensitivity and separately labeled tail/parity diagnostics
  are retained. No reflecting/clipping state boundary.
- Happiness bandits: four environments,2000rounds×5000actions,38000condition-round
  records,190million choices. Sudden fixed−objective+370.82 matches published
  +370.11 closely; stationary-spread fixed−objective−95.60 reverses published
  +264.50. Constant-change dynamic advantage+2582.93. Fixed aspiration and an
  ordinary matched optimistic learner have zero disagreements over40million
  paired actions. Original Supplement S1's gridworld optimism comparison was
  checked and cited; it was not executed or refuted by this scoped equivalence.
- Chain-of-Emotion original-data reanalysis: author-score counts24/31/35 of42
  reproduce exactly. Numeric condition trends closely recover LIWC F/p patterns;
  categorical analyses differ. All30participants/90ratings retained; published
  questionnaire means/df do not reproduce exactly. GG+Holm12 retainsnaturalness;
  an additional explicitly posthoc ordinal Friedman+Holm12 retains3items.
  Methods-dependent results remain separate; no new participants or GPToutputs.
- EmotionPrompt:64realQwencalls/32pairs. Original benchmark accuracy19/32→22/32,
  difference+.09375,pairedCI[-.03125,.21875],exactMcNemarp=.453125. Correct equation
  outputs explain a formatting confound: exploratory arithmetic content12/16 in
  both conditions. Primary scoring unchanged. No convincing ability improvement.
- Independent record-only audit recomputes raw means/SD/counts and scores without
  model/simulation calls. Separate upstream-scoring, full-horizon deterministic
  bandit recheck and per-study source/artifact checks pass. All173Python tests
  pass; compileall passes. Historical native code did not change.
- Main comparison PNG/SVG is `results/papers/figures/paper_comparisons`.
  Independent critic report: `docs/reproductions/review.md`; numeric checks:
  `results/papers/reviewer_checks.json`. Global hashes are in
  `results/papers/provenance.json`. Draft PR1 is updated, not merged.
