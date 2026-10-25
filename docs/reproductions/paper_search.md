# Paper selection and reproducibility search

Search date: 2026-10-25. This bounded search prioritised primary papers and
author repositories with equations, fixed tasks, numerical tables or original
outputs. Feasibility depends on the CPU-only environment and the already
cached open-weight Qwen2.5-Coder-1.5B-Instruct model, rather than assuming
access to historical paid APIs or new human studies.

| Paper | Publication and sources checked | Feasible reproduction | Main limit |
|---|---|---|---|
| Keramati & Gutkin, **Homeostatic reinforcement learning for integrating reward collection and physiological stability** | [eLife 2014](https://doi.org/10.7554/eLife.04811), full article and supplementary parameter tables | Independently implement Figure 6 at the original 100,000-agent cohort scale and million-action horizon | Original executable code/RNG unavailable; softmax convention unresolved; no invented figure endpoints |
| Dubey, Griffiths & Dayan, **The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons** | [PLOS Computational Biology 2022](https://doi.org/10.1371/journal.pcbi.1010316), full paper and [author code](https://github.com/rach0012/happiness_RL) | Independently implement Experiments 2a/2b at 2,000 rounds × 5,000 actions; compare published cumulative-reward means and SDs | Released code covers gridworlds; bandit seeds/tuning runner and one aspiration schedule unavailable |
| Croissant et al., **An appraisal-based chain-of-emotion architecture for affective language model game agents** | [PLOS ONE 2024](https://doi.org/10.1371/journal.pone.0301033), full article/XML, [OSF original CSVs](https://doi.org/10.17605/OSF.IO/QPT6Z) | Re-execute all 42 STEU scores, released LIWC analyses, and participant-matched analysis of 30×3 user ratings | Statistical reproduction of existing observations, no fresh participants/GPT-3.5; LIWC dictionary not rerun; public dataset has no explicit OSF license |
| Sabour et al., **EmoBench: Evaluating the Emotional Intelligence of Large Language Models** | [ACL 2024](https://aclanthology.org/2024.acl-long.326/), [paper](https://aclanthology.org/2024.acl-long.326.pdf), [author code and MIT data](https://github.com/Sahandfer/EmoBench) | Public 200 EU and 200 EA tasks, official prompt templates and scorer; full random/majority baseline can run cheaply | Original model experiment needs 5 samples ×4 orderings, with joint emotion/cause EU accuracy; the previous JSON-answer pilot is not the published protocol |
| Li et al., **The Good, The Bad, and Why: Unveiling Emotions in Generative AI** | [ICML 2024](https://proceedings.mlr.press/v235/li24bs.html), earlier [EmotionPrompt preprint](https://arxiv.org/abs/2307.11760), [author project repository](https://github.com/llm-enhance/llm-enhance.github.io) | Emotional-prompt versus neutral-prompt comparison on published tasks with a local model and paired uncertainty | Historical model versions and full task/model scale differ; appended emotional language is not persistent affect |
| Lee et al., **Life-inspired interoceptive AI** / EVAAA | [primary UCL record](https://discovery.ucl.ac.uk/id/eprint/10230924/), [official EVAAA Unity/Python code](https://github.com/cocoanlab/evaaa) | Inspect environment and homeostatic variables; runnable authors' training code exists | Published Perspective is not a clean tabular CPU experiment; Unity plus long RL training was not selected for a claimed reproduction |

EmoBench was a serious candidate because its source data and prompt interface
are public. The original protocol is more demanding than a single keyed exam:
it samples five model answers and majority-votes each item under four choice
orderings. EU jointly checks an emotion and its cause. Table 1 reports English
EU accuracy 22.50% for Qwen-Chat-7B and 59.75% for GPT-4; Table 2 reports English
EA accuracy 54.62% and 75.50% respectively. Those are published context values,
not results reproduced by regesh. The official repository tree contains code
and data but no archive of those model outputs was located in this search.

Chain-of-Emotion was selected instead as the core empirical affect-paper
reproduction because its released observations permit exact score auditing
and meaningful sensitivity analyses. Independent implementations of
homeostatic reinforcement learning and comparison/aspiration experiments
are documented in their own methods files. A lower-cost live-model emotional
prompting replication adds a different evidence type. Multiple experiments
within one paper do not increase the count of distinct reproduced papers.

Terms such as replication, statistical reproduction and conceptual adaptation
are kept separate in the combined report. A numerical match to authors' raw
outputs does not reproduce data collection; reproducing a mathematical model
does not establish that its emotion interpretation is unique or biologically
accurate.
