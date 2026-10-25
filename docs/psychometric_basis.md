# Primary research behind the expanded measurement design

Search date: **2026-10-25**. This is an implementation-oriented check of primary
papers/venue records, not a systematic psychometric review. Source results below
were not independently reproduced. The present project has no human participant
study and does not copy proprietary instruments.

| Primary source | Evidence type and relevance | Consequence for Regesh |
|---|---|---|
| [EmoBench: Evaluating the Emotional Intelligence of Large Language Models](https://aclanthology.org/2024.acl-long.326/) | Peer-reviewed ACL2024 benchmark; distinguishes emotional understanding and application rather than recognition alone | Use externally keyed questions from both domains. Static answers assess capability; histories/interventions assess persistent regulation separately. |
| [A psychometric framework for evaluating and shaping personality traits in large language models](https://www.nature.com/articles/s42256-025-01115-6) | Peer-reviewed Nature Machine Intelligence2025; reports reliability and validity evidence for output personality measurements under particular model/prompt configurations, using multiple validation approaches | Personality-response studies are possible, but require convergent, discriminant and behavioral criterion evidence. A consistent questionnaire response alone does not establish a human-like trait or feeling. |
| [On the Reliability of Psychological Scales on Large Language Models](https://aclanthology.org/2024.emnlp-main.354/) | Peer-reviewed EMNLP2024; studies many prompt/settings variations and response consistency | Test ordering/context effects explicitly. Reliability and construct validity are separate questions. Our twelve-item pilot is too small for a general psychometric-validity claim. |
| [Large language models display human-like social desirability biases in Big Five personality surveys](https://pmc.ncbi.nlm.nih.gov/articles/PMC11650498/) | Peer-reviewed primary study of survey-context-dependent response shifts; effects persist under some order/paraphrase changes | Include persona/context sensitivity and preserve a keyed task endpoint. Do not treat socially desirable answers as universally correct personality answers. |
| [Large language models are proficient in solving and creating emotional intelligence tests](https://www.nature.com/articles/s44271-025-00258-x) | Peer-reviewed Communications Psychology2025; evaluates ability-style emotion tests and generated items | Strong static EI answers can coexist with the absence of any tested persistent external controller. Compare useful behavior, not only test fluency; avoid circularly validating generated items with the same model. |
| [Emotional intelligence of Large Language Models](https://journals.sagepub.com/doi/full/10.1177/18344909231213958) | Peer-reviewed2023 SECEU work; compares response patterns with a human reference distribution | Distributional agreement can complement exact keys when licensing and reference data permit. We do not have participant response norms sufficient to reproduce it here. |

These sources motivate a battery rather than one score: keyed accuracy, probability
calibration, abstention/coverage, semantic consistency under option order, sensitivity
to persona, actual coping/control under changed goals, and independently verified
useful solutions. Human inventories without correct answers need a separate
expression/trait protocol; their scores cannot replace useful-agent evaluation.
Emotional vocabulary, model self-report, psychological-test knowledge, persistent
regulation, development, biological fidelity and subjective experience remain
separate constructs.

The present QA pilot uses published author keys and preserves normative ambiguity,
language/cultural limits and possible pretraining exposure as limitations. Its
secondary choice-only score is labeled separately from strict structured-response
scoring and proper probability scores. No probability distribution is invented
when a model fails to supply a valid one.
