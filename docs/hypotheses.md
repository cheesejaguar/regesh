# Mechanism competition and independent critique

Architecture-selection memo, searched **2026-10-25**. This document records hypotheses before implementation; the implementation and final report determine which components were actually built and measured. The independent hypotheses reviewer did not implement the platform or choose its results. None of the candidates below establishes subjective experience or biological fidelity.

## Checked antecedents and scope

Primary records checked for the decisions below:

- **[ALMA (Gebhard, AAMAS 2005)](https://www.dfki.de/en/web/research/projects-and-publications/publication/1782/)**: short, medium, and long affective layers are established precedent. The institutional abstract, rather than its complete implementation, was checked.
- **[EMA (Marsella and Gratch, Cognitive Systems Research 2009)](https://www.sciencedirect.com/science/article/pii/S1389041708000314)**: appraisal, interpretation, and coping in a changing cognitive model are established precedent. Publisher abstract checked; the supplied author PDF URL failed in the browser. Do not claim experience-dependent appraisal or counterfactual coping in general as new.
- **[Homeostatic RL (Keramati and Gutkin, eLife 2014)](https://elifesciences.org/articles/04811)**: internal-state-sensitive value and anticipatory regulation are established precedent, supported by formal analysis and simulations. Operational computational resources are an engineering analogy, not evidence about physiology.
- **[Intelligent Adaptive Curiosity (Oudeyer, Kaplan, and Hafner, IEEE TEC 2007)](https://www.pyoudeyer.com/ims.pdf)**: region-dependent progress in prediction is an established exploration objective, with simulated and physical robot demonstrations. Maximizing raw surprise can trap an agent in noise; progress must be distinguished from error magnitude.
- **[Strategy selection as rational metareasoning (Lieder and Griffiths, Psychological Review 2017)](https://doi.org/10.1037/rev0000075)**: choosing cognitive strategies according to benefits and computation costs is established precedent. Author publication record and indexed abstract checked, not a reproduction of its experiments.
- **[Successor Features for Transfer in RL (Barreto et al., NeurIPS 2017)](https://papers.nips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html)**: separating predicted feature consequences from preferences supports transfer when rewards change with stable dynamics. Applying the factorization to operational needs is a possible synthesis, not a new general transfer principle.
- **[A Retrieved Context Model of the Emotional Modulation of Memory (Talmi, Lohnas, and Daw, Psychological Review 2019)](https://www.repository.cam.ac.uk/items/2e79efd8-b2e2-4c3a-8888-0d1ca37a3de8)**: emotional encoding/retrieval context is prior work. Institutional/indexed abstract checked; no quantitative replication performed here.
- **[On a class of dual control learning systems (Feldbaum, 1964)](https://www.mathnet.ru/php/archive.phtml?jrnid=at&option_lang=eng&paperid=11622&wshow=paper)**: actions can jointly regulate a system and reveal its dynamics. Catalog abstract checked. Generic counterfactual information-aware action selection is not a defensible novelty claim.

These checks bound the novelty claims rather than establish an exhaustive priority search. See the research matrix for newer agent implementations and additional verification gaps. Source summaries are deliberately short; no paper was reproduced by this reviewer.

## Candidate families

The competition spans **A: explicit appraisal/dynamical control**, **B: predictive/learned metacontrol**, **C: memory/credit representations**, and **D: ambiguity/robust inference**. C7 and D8 were independently introduced by this review rather than named in the task's example directions. Their component ideas still have antecedents.

### A1 — Layered target-specific appraisal reference

- **Change:** map evidence to goal relevance, gain/loss, surprise, responsibility, uncertainty and controllability; update separate target-specific fast components and slow moving averages. Relationships separately track familiarity, affiliation, factual reliability, and authority. Derive bounded retrieval/verification/exploration controls from these states.
- **Antecedent/departure:** ALMA's temporal layers and EMA's appraisal/coping are direct antecedents. Event provenance, replay, independent evidence retrieval, and operational needs make this an inspectable engineering synthesis; the layering is not new.
- **Signature:** identical observations produce different regulation under changed accepted goals or control; mixed target states coexist; affect decays while learned expectations survive.
- **Minimum implementation:** deterministic typed appraiser, exponential transitions, persistent target records, explicit control equations, SQLite transaction journal.
- **Experiment/falsifier:** replay one history under changed goals and intervene on mood with observations fixed. If only language changes, or control outputs are unaffected, the claimed functional role fails. If recall produces the same expectation learning as fresh external evidence, the causal design fails.
- **Data/cost/confounds:** reviewed fixtures; negligible CPU cost. Handwritten mappings can encode the intended answer; this validates mechanics, not human emotion or creativity.

### B2 — Counterfactual regulation of operational viability

- **Change:** learn action-conditioned predictions of quality, remaining resources, uncertainty, commitments, and reliability. For alternative cognitive actions, project the next internal state and select by expected goal quality minus operational deficit costs. Fast affect supplies a bounded context-sensitive residual, while feasibility remains a hard constraint.
- **Antecedent/departure:** homeostatic RL, model predictive control, dual control, and rational metareasoning already predict consequences and regulate resources. A persistent, provenance-aware coupling to learned appraisals and memory could be a new tested synthesis; an affect label adds no novelty.
- **Signature:** anticipatory restoration or verification occurs before a hard deficit, but high load with ample capacity need not produce aversion. Regulator switches from divergent search to verification when evidence shows a plausible candidate or deadline.
- **Minimum implementation:** three or four actions (`explore`, `verify`, `commit`, optionally `restore`), a linear context/action outcome predictor, one-step projection, and logged candidate scores.
- **Experiment/falsifier:** held-out contexts with the same immediate reward but different delayed resource consequences; ablate projection while matching features and extra arithmetic. Equal or worse goal quality, failed viability, or no measurable behavioral difference counts against the mechanism's utility.
- **Data/cost/confounds:** simulator action/outcome records, trained only on declared training families; CPU least squares or small SGD. A simulator can favor its own planner, and internally assigned costs may dominate externally checked quality. Audit both and report independent task metrics.

### B3 — Learned distributed modulation receptors

- **Change:** share an appraisal state but learn separate small residual maps for memory weighting, exploration, verification, learning rate, and horizon. Fit each map to observed marginal benefits rather than demand that one scalar changes every process in the same direction.
- **Antecedent/departure:** neuromodulatory specialization and learned cognitive strategy control are precedents. The prospective departure is a transparent, separately intervenable action-interface synthesis with agent-specific adaptation, not an assertion that its numbers are hormones.
- **Signature:** the same uncertainty can increase verification in brittle tasks and exploration in learnable tasks; a control intervention on one receptor leaves the other heads unchanged.
- **Minimum implementation:** feature vector with normalized observations and fast/slow states; linear heads with bounded residuals; train on actual simulated checks. No RNN is necessary to test this hypothesis because persistence is explicit.
- **Experiment/falsifier:** compare one shared modulation coefficient with separate heads under equal parameter count (allocate matched unused/general features if necessary). If held-out effects vanish, or parameter count explains gains, distributed modulation lacks support.
- **Data/cost/confounds:** logged outcomes with exploration coverage; small CPU regression. Selection bias makes action values unidentifiable when the training policy never attempts alternatives. Randomize actions in training or enumerate simulator actions, and disclose simulator supervision.

### C4 — Evidence-linked episodic reappraisal graph

- **Change:** store immutable event evidence and separately version inferred meaning. Link events to targets/goals/actions/outcomes. New contradictory evidence adds a new appraisal revision; retrieval can reinstate bounded affect without incrementing factual evidence or relationship counts.
- **Antecedent/departure:** emotional context retrieval and cognitive appraisal already interact with memory. Separating immutable evidence from interpretable revisions is engineering practice; joint causal replay of appraisal development is a useful research facility rather than an established new cognitive algorithm.
- **Signature:** old criticism can be reinterpreted as accurate help after an independently checked repair; future criticism by that source receives changed reliability weighting. A quotation or recall cannot fabricate another failure.
- **Minimum implementation:** relational graph via causal IDs, appraisal revision table, evidence-focused and affect-biased retrieval routes, capped reactivation, deduplication.
- **Experiment/falsifier:** introduce disconfirming evidence with the same old event retained. If future appraisal does not change, development is absent; if factual event counts change, evidence and interpretation were conflated. Sustained mood/retrieval loops must recover after a perturbation.
- **Data/cost/confounds:** small reviewed event histories; storage/retrieval CPU only. Mood congruence can create self-confirming retrieval; measure factual recall and stability alongside apparent coherence.

### B5 — Successor features of coping consequences

- **Change:** predict discounted future feature occupancy for strategies, `psi(s,a)=E[sum gamma^t phi(s_t)]`, then reevaluate using goal/internal-need weights `w`; update successor features by temporal-difference error.
- **Antecedent/departure:** successor features and generalized policy improvement directly supply the computation. Using operational need features and a slowly developing appraisal weighting could be a synthesis, with little algorithmic originality.
- **Signature:** transfer when task goals or resource prices change but the action dynamics remain familiar; weaker transfer when transition dynamics change.
- **Minimum implementation:** tabular feature successor values for a few strategies and a state abstraction; no neural model required.
- **Experiment/falsifier:** separate reward revaluation from transition revaluation. If transfer is no better than relearning, or if an alleged transfer result uses the same task with changed words, evidence is weak.
- **Data/cost/confounds:** many short simulated trajectories; low CPU. State abstraction can leak task identities, and unchanged dynamics make transfer artificially easy.

### B6 — Learning-progress divergent/convergent scheduler

- **Change:** estimate local improvement in prediction or verified candidate quality over two windows. Allocate search to regions with positive expected progress and verification to promising candidates; stop escalating exploration in irreducible noise.
- **Antecedent/departure:** Intelligent Adaptive Curiosity and rational metareasoning closely anticipate this. Persistent appraisal can smooth strategy changes; neither progress motivation nor exploration/verification switching is new by itself.
- **Signature:** habituation to solved cases, abandonment of noise traps, renewed exploration after a learnable change, and later convergence on useful designs.
- **Minimum implementation:** per-region prediction-error windows, progress estimate, bounded mode-selection score.
- **Experiment/falsifier:** mix solved, learnable, and noisy task regions. Compare against uncertainty-only exploration with identical calls. If time spent in noise or verified quality does not improve, the proposed scheduler benefit fails.
- **Data/cost/confounds:** outcome histories; low CPU. Apparent progress can reflect easier samples rather than learning; use fixed evaluation probes and a held-out region.

### C7 — Causal credit escrow (independent candidate)

- **Change:** keep evaluative claims in an uncertainty-weighted ledger until observed outcomes resolve causal credit. Praise may update affiliation but cannot update competence or task success. Counterfactual action credit is assigned only from admissible evidence or explicitly labeled simulator access.
- **Antecedent/departure:** delayed credit assignment, Bayesian evidence, and provenance systems are antecedents. The proposed departure is affective updating through an explicit, revisable credit ledger; novelty is unverified and no first-in-literature claim is justified.
- **Signature:** praise without success increases social warmth without increasing competence expectations; accurate unpopular criticism can improve factual reliability. Unresolved commitments influence verification without being treated as failures.
- **Minimum implementation:** pending prediction/outcome links with resolution state, separate competence/social update channels, revision records for later corrections.
- **Experiment/falsifier:** adversarial praise, misleading criticism, delayed verification, and source-identity reversal. If unsupported social feedback changes goal success or authority, the separation fails. If the ledger improves no calibration or task metric, its complexity is not justified.
- **Data/cost/confounds:** deterministic paired histories plus externally checked outcomes; low CPU. Delayed learning can reduce responsiveness; a strong ordinary evidence tracker may obtain every benefit without affect.

### D8 — Set-valued ambiguity-preserving appraisal (independent candidate)

- **Change:** retain multiple plausible interpretations of ambiguous intent and compute an interval or distribution of appraisal/control consequences. Choose low-regret evidence-gathering actions until discriminating information arrives, rather than collapse hostile/helpful intent to one LLM story.
- **Antecedent/departure:** Bayesian belief-state control, robust decision making, and EMA's interpretation dependence are close antecedents. Applying explicit ambiguity sets to affect trajectories is a testable synthesis, not a new uncertainty principle.
- **Signature:** ambiguous criticism initially produces mixed states and proportionate verification; later evidence narrows interpretation without changing the stored quote.
- **Minimum implementation:** two or three intent hypotheses with probability bounds and evidence IDs; expected or worst-case bounded control; observations update hypothesis weights.
- **Experiment/falsifier:** pair histories with identical ambiguous prefix and different later intent evidence. Compare single-label appraisal against the ambiguity set with equal context size. No calibration improvement, chronic indecision, or failure to revise counts against it.
- **Data/cost/confounds:** paired reviewed histories; low CPU. Hypothesis generation can omit the truth; more context or computation alone could improve results. Avoid giving robust control privileged information.

### A9 — Change-point-dependent affect timescales

- **Change:** infer environmental volatility from prediction residuals and adjust slow/fast filter weighting; stable relationship histories resist isolated errors while sustained regime shifts trigger faster adaptation.
- **Antecedent/departure:** adaptive filtering, hierarchical uncertainty estimation, and multiple affect timescales precede this. The departure would be a learned, explicitly tested timebase for target-specific coping rather than one fixed mood decay.
- **Signature:** habituation during repeated harmless signals, recovery after interruption, and rapid evidence-driven change after a true partner reliability reversal.
- **Minimum implementation:** two exponential filters and a bounded residual-based change-point score; optional learned mixing coefficient.
- **Experiment/falsifier:** matched stationary/noisy and abruptly changing partners. If a fixed-rate ordinary estimator is equally calibrated and useful, the adaptive timescale has no demonstrated advantage.
- **Data/cost/confounds:** short synthetic longitudinal traces; low CPU. Faster learning after an engineered reversal may simply match the generator's timescale; vary reversal rate in held-out tests.

### D10 — Affect as a minimal predictive control summary

- **Change:** train a small latent summary of history to predict which future cognitive interventions will improve verified quality, penalizing state size. Compare it with named affect coordinates and an unconstrained recurrent ordinary controller.
- **Antecedent/departure:** predictive-state representations, information bottlenecks, and learned recurrent control are antecedents. A measured minimal sufficient state for affect-like metacontrol could clarify representation choice; naming the state emotional contributes no evidence.
- **Signature:** a compact summary preserves transfer-relevant causal history while omitting redundant surface descriptions; intervention on its coordinates should change controls predictably.
- **Minimum implementation:** linear low-rank history encoder and prediction heads; first fit outcomes, then add dimensional penalty. Defer neural recurrent versions until simple models fail.
- **Experiment/falsifier:** equal-capacity latent, explicit appraisal, and ordinary summaries on held-out families and event orders. If explicit coordinates or raw running statistics match it, compression adds no demonstrated utility. If interventions are uninterpretable or unstable, the inspectability objective fails.
- **Data/cost/confounds:** substantially more independent trajectories than A1/B2; still CPU-scale for linear models. Latent coordinates may encode task IDs; apparent compression can hide unmeasured information access.

## Ranking before implementation

Scores range 1–5. Utility means plausible contribution under limited CPU/data; originality is **relative to the checked sources only**, not established priority. Equal weights produce the mean; implementation choices should favor a coherent loop rather than mechanically maximize this table.

| Candidate | Family | Utility | Originality | Testability | Coherence | Feasibility | Mean | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 reference | Explicit dynamics | 4 | 1 | 5 | 5 | 5 | 4.0 | Required transparent core |
| B2 counterfactual regulation | Predictive control | 4 | 2 | 5 | 5 | 4 | 4.0 | Preferred experimental mechanism |
| B3 distributed residuals | Learned metacontrol | 4 | 2 | 5 | 4 | 4 | 3.8 | Preferred small learned extension |
| C4 episodic revisions | Memory representation | 4 | 2 | 5 | 5 | 4 | 4.0 | Core facility; limit novelty claim |
| B5 successor coping | Predictive transfer | 3 | 2 | 4 | 4 | 3 | 3.2 | Defer: extra state abstraction |
| B6 progress scheduler | Learned metacontrol | 4 | 1 | 5 | 4 | 5 | 3.8 | Use as antecedent/baseline |
| C7 credit escrow | Causal credit | 4 | 3 | 5 | 5 | 5 | 4.4 | Use evidence separation; defer novelty |
| D8 ambiguity sets | Robust inference | 4 | 3 | 4 | 4 | 4 | 3.8 | Preserve alternatives in core |
| A9 adaptive timescales | Explicit dynamics | 3 | 2 | 5 | 4 | 4 | 3.6 | Defer learned version |
| D10 minimal summary | Representation learning | 3 | 3 | 3 | 3 | 2 | 2.8 | Defer until data justify it |

## Selected coherent synthesis

Use A1 + C4 with C7's separation of success from praise and D8's retained alternatives as the transparent reference. These make persistence, development, provenance, and causal tests meaningful. Implement **B2** as the main experimental mechanism and **B3** as a small learned component only if training data cover its action choices. Keep one-step counterfactual control and linear heads first; this avoids requiring opaque recurrence merely to claim a learned architecture.

An illustrative numerical specification (the final code may refine it):

1. `x` contains goal relevance, observed progress, uncertainty, controllability, remaining budget, unresolved commitments, independent factual reliability, and explicit fast/slow state. Source identity and task-family ID are excluded from the shared learned predictor.
2. For cognitive action `a`, fit `y_hat(a)=W_a x` using training-only simulator observations or randomized action/outcome logs. Targets are verified quality change and resource/uncertainty consequences, not emotion names written by the same LLM.
3. Score alternatives by verified-goal expectation minus projected operational deficit penalties plus bounded head-specific state residuals. Hard feasibility constraints apply before ranking. Save the alternative predictions and score components as structured reasons.
4. On a verified outcome, adapt agent-specific expectations with a declared update rule. Keep generic fitted `W` fixed during held-out evaluation. Agent adaptation, appraisal history, and all rolling statistics live in the trial snapshot.
5. A disposition may develop from accumulated supported predictions, while a recent appraisal still has short-lived mixed components. No scalar mood is a proxy for truth, authority, or reward.

Ridge regression or a tiny NumPy SGD model is sufficient on this problem scale. Explicit state supplies memory, so the selected experiment does not need an RNN and makes no recurrent-learning claim. If the trained component is a policy head rather than a transition predictor, report that change plainly and do not imply that a learned world model was implemented.

## Discriminating experiment and controls

The primary causal hypothesis is: **history-dependent operational appraisal improves switching between exploration and verification under a fixed budget**. The behavioral endpoint is executable constraint satisfaction and verified solution quality, with originality relative to a fixed benchmark reference corpus reported separately.

Train on two families with varied partner reliability, controllability, resources, and setbacks; choose hyperparameters on a development family or partition; freeze shared weights before evaluating an unseen task family. Hold out partner identities and event orders. Give each condition identical observations, candidate generation access, checks, calls, tokens, and time limits. Count appraisal/reflection calls as expenditure. Begin every trial from a declared full snapshot; no adaptation from one test trial reaches another.

Required conditions are: memory/planning only, static persona, turn-local conditioning, persistent heuristic, selected hybrid, and an ordinary adaptive controller. Two additional distinctions are essential:

- A **no-label clone of the hybrid** removes emotion vocabulary while leaving every numeric feature and action unchanged. Deterministic numerical runs should be identical; any difference indicates labels reached the behavior or a reproducibility error. This tests labels, not architecture.
- The **structurally different ordinary adaptive controller** must receive the same history-derived information and controls with similar parameter capacity, training data, and budget. A running estimator plus contextual action-value policy is a stronger comparison than a stateless greedy baseline. Its label-free structure is allowed to exploit every accessible predictor.

Use frozen persistent state, within-split shuffled state (no cross-trial leakage), disconnected affect-to-memory weights, removed counterfactual projection, and removed learned residuals. Report if an intervention changes the information given to the controller as well as state. A disconnected-memory condition must preserve the same retrieval budget and an evidence path. Evaluate sustained perturbations and return-to-baseline dynamics; bounded scalars do not prove stability.

A small CPU run can use fixed synthetic design grammars for systems optimization, scientific experiment choice, and transfer of a solution primitive under a new constraint. Count mechanism identities rather than lexical diversity. These are **simulator/controller experiments**, even when the design actions have engineering names. A later live-model experiment should substitute the same base model and blind independent checking without changing the controller protocol.

## Independent adversarial critique

1. **The main novelty story is weak without a specific departure.** The combination of appraisal, persistent mood, learning, and memory was anticipated by cognitive-affective architectures. Homeostatic RL already predicts internal-state consequences; metareasoning already schedules computation. The strongest defensible present claim is an inspectable engineering synthesis plus a falsifiable experimental mechanism.
2. **Affect may be a renamed sufficient statistic.** If an ordinary adaptive controller with the same information matches the hybrid, emotion terminology explains nothing additional. This is an informative result and should be prominent. Removing labels from identical code cannot establish that affective dynamics are unnecessary; it only establishes that the vocabulary is unnecessary.
3. **A fixture can be solved by the author.** Handwritten appraisal examples validate invariants. Synthetic tasks whose reward function exactly matches the affect score can demonstrate that the implementation follows its objective, not that affect improves useful ideas. Test independent endpoints and multiple generator variants.
4. **History is a major confound.** A persistent method beats a turn-local comparator partly because it has memory. Give the ordinary adaptive baseline all the same raw history and compact estimates. Match context size by padding or providing equivalent structured summaries without extra task facts.
5. **Training labels are not ground truth just because they are numerical.** A simulator quality score can encode the designer's preferences. Prefer independent feasibility checks, held-out objective functions, and outcome-derived supervision. LLM-generated emotional labels are particularly circular.
6. **Counterfactual predictions can receive privileged access.** Enumerating all simulator action outcomes is acceptable for training if disclosed. At test time the controller must not inspect outcomes it has not purchased or executed. Every condition must pay for the same information.
7. **A positive creativity score can reflect search coverage.** More diverse candidate mechanisms are useful only when feasible and supported. Report original-but-useless and useful-but-common separately, and compare at matched total tool/token/check budgets.
8. **Long-term development needs more than counters.** A new event after reliability learning must appraise differently with its evidence held fixed. Reinterpreting an old event must retain the evidence and version the interpretation. Cross-trial contamination can otherwise imitate development.
9. **No inference to experience is justified.** Persistent functional regulation does not establish feeling, phenomenal consciousness, psychological validity, or biological fidelity. Those require different theories and evidence.
10. **A negative result must change the narrative.** If gains disappear against ordinary adaptive control, say that the platform works but architectural affect has no demonstrated advantage in these tasks. Do not replace the failed comparison with favorable persona comparisons.

## Claim ledger at selection

| Claim | Label | Evidence required or available |
|---|---|---|
| Appraisal/coping and multiple affect timescales have prior implementations | Established precedent | EMA and ALMA primary records |
| Predictive internal regulation and progress-driven exploration have prior formalizations | Established precedent | Homeostatic RL and IAC |
| Immutable events plus revisable appraisal and replay form a useful platform | Engineering synthesis | Implementation and invariant tests |
| Counterfactual operational affect and distributed residuals improve control | Speculative hypothesis | Matched held-out outcomes and ablations |
| The proposed architecture is first or broadly novel | Unsupported; not claimed | Exhaustive closest-art review absent |
| Affect improves useful original live-model ideas | Speculative hypothesis | Same-model budget-matched live evaluation absent at selection |
| The agent has subjective feelings or biological emotional fidelity | Unsupported; not claimed | No relevant measurement |

Measured results belong in the final report with run configurations and raw artifacts; none are asserted by this architecture-selection memo.
