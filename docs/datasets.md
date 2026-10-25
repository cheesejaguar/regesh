# Dataset inventory and measurement map

Search and inspection date: **2026-10-25**. This bounded review catalogs **57 resources** from author repositories, official dataset cards, project pages and primary papers. It is not an exhaustive frontier search. Publicly downloadable, open-source code, open-data licenses, noncommercial research access, and proprietary psychometric instruments are different categories. The table preserves those distinctions. A repository's MIT/GPL/Apache license does not automatically license underlying television clips, exam questions, platform text or participant records.

The machine-readable [catalog](../datasets/catalog.json) records source links, inspected card revisions where available, access, license evidence, limitations, intended use and metrics. Primary metadata was inspected for the listed repositories/cards; only four datasets were actually downloaded. A card/repository declaration is useful evidence of published terms, not a legal guarantee of upstream rights. Rows marked review/restricted are **not** part of the automated download allowlist.

## Immediate recommendations

1. **EmoBench** is the best small starting point for the user's psychology-question/answer idea: emotional understanding and effective action choices, with an author MIT license. Score the provided answer key rather than asking the generating model to judge itself. English pilot IDs and sources are frozen in `datasets/emobench_pilot_source.json`; those items must remain evaluation-only. Passing these static questions shows emotional-task knowledge/ability under this format, not persistent agent affect.
2. **GoEmotions** supplies independently annotated multilabel emotion-recognition data. Its official card/repository declare Apache-2.0. Separate its training and test splits, preserve original IDs, and report recognition rather than claiming appraisal or development.
3. **HumanEval/MBPP plus EvalPlus tests** give executable correctness, robustness and repair checks. HumanEval is MIT; MBPP's data card says CC-BY-4.0. These familiar public tests may be in model pretraining, so use fresh held-out task families and independent extra tests for the causal comparison. Do not tune a controller on its final evaluation tasks.
4. **LongMemEval/PersonaMem** provide targeted later work on stale memories, updates, abstention, time and preferences. LoCoMo is also useful, but respect its CC-BY-NC-4.0 terms and cluster uncertainty by conversation rather than treating correlated questions as independent participants.
5. **HelpSteer2** provides human ratings of helpfulness, correctness, coherence, complexity and verbosity. It can support quality-rubric validation, but preference ratings cannot replace executed task outcomes.
6. **IPIP** is a genuinely public-domain personality item pool. It is useful for test/retest and context sensitivity. There is no answer key of psychologically correct personality responses. The clinical PsychBench dataset is gated/restricted; MSCEIT and commercial diagnostic/personality forms are not substitutes for open data.

## Downloaded evidence and reproduction

Executed, size-and-SHA-256-verified downloads total **5,213,911 bytes in nine artifacts across four datasets**: EmoBench EA/EU and license; GoEmotions train/test and label names; HumanEval corpus and license; MBPP corpus. Actual caches are outside the repository at `/workspace/scratch/regesh-datasets`. The small EmoBench evaluation fixture is separately attributed with its original MIT notice in `datasets/licenses/emobench-LICENSE`. No model weights were trained on these downloads in this work.

```bash
python scripts/fetch_datasets.py --list
python scripts/fetch_datasets.py --cache /workspace/scratch/regesh-datasets
python scripts/fetch_datasets.py --cache /workspace/scratch/regesh-datasets --verify-only
# Or fetch only the reviewed external QA source:
python scripts/fetch_datasets.py --cache ~/.cache/regesh/datasets --ids emobench
```

The downloader uses commit-pinned HTTPS URLs, exact expected byte lengths and hashes. It never imports dataset scripts or executes reference solutions, never requests credentials, and refuses conflicting preexisting cache contents. `fetch-provenance.json` records verified paths, source revisions, hashes, licenses and retrieval time. Restricted/gated/unclear rows have no automatic URL. These downloaded files are source material, not measurements of model quality. Dataset loading must not leak answer labels into model prompts.

## Psychology: three different questions

**Knowledge/ability:** EmoBench, SocialIQA and MMLU psychology subjects have gold choices. Report accuracy with chance baselines, category breakdowns, invalid-answer rates, option permutations, paraphrases, confidence calibration when meaningful, and uncertainty clustered by independent scenario. Emotional recognition, counseling-exam knowledge, theory of mind, and practical social success are different constructs.

**Expression/traits:** IPIP questionnaires describe response tendencies rather than correct answers. Reverse-key consistency, test/retest reliability and prompt/context sensitivity are useful. A human norm percentile, diagnosis, or claim that the model has a stable human trait is unsupported without measurement invariance and external validation. An LLM may know socially desirable answers and respond as a requested persona. Never score an inventory as if maximum extraversion or minimum neuroticism were task success.

**Persistent functional regulation:** static exam performance cannot show this. Pair the same later item with different verified histories, accepted goals and internal resources. Test whether changes track supported evidence, disappear under frozen state, and influence actual retrieval/planning/actions. Include accurate versus misleading criticism, praise without success, disconfirming evidence, memory neutral-path access, and delayed repair. Keep source evidence separate from generated interpretation.

The similarly named resources **CUHK PsychoBench** (personality/questionnaire tooling), **PsychCounsel-Bench/PsychoBench** (psychology exam QA), and **BMEr-ATP PsychBench** (clinical records) should not be merged. The catalog includes the latter two with their restrictions. Commercial MSCEIT/MMPI/copyrighted BFI/NEO materials are not automatically reusable because a paper describes them; IPIP offers a public-domain alternative for expression studies.

## Many complementary quality measures

| Dimension | Independently checkable measurement | Main confound / interpretation limit |
|---|---|---|
| Correctness | Hidden test pass rate, pass@1, regression count | Familiar test memorization; insufficient oracle coverage |
| Robustness | Property/metamorphic tests, mutation score, boundary cases | Poorly chosen properties can encode the author's solution |
| Feasibility | Explicit memory/time/tool/engineering constraint compliance | Internal self-assigned satisfaction is not external feasibility |
| Useful originality | Mechanism novelty against substantive fixed corpus, conditional on correctness | Absence from a tiny corpus is easy; semantic distance is insufficient |
| Diversity | Distinct valid algorithms/mechanisms and quality-diversity frontier | Wording differences and random invalid proposals inflate naive diversity |
| Revision | Paired before/after quality after an actual new constraint or accurate criticism | A general second attempt can improve without affect |
| Transfer | Family/API/partner/time-held-out utility and constraints | Shared grammar or leaked identities make transfer too easy |
| Calibration | Brier, log score, reliability diagrams, risk-coverage | Self-reported probability may be unfaithful; do not confuse a shared predictor's calibration with policy quality |
| Evidence support | Unsupported-claim rate, citation entailment, source-ID precision | An LLM judge can share the generator's errors |
| Appraisal | Goal-conditioned accuracy/MAE, supported alternatives, annotator agreement | Human subjective interpretation is not a single uncontested gold label |
| Memory accuracy | Recall precision/recall, temporal order, stale-fact error, abstention | Answering from model pretraining can mask retrieval failures |
| Memory efficiency | Stored bytes/event, RSS, ingestion/retrieval p50/p95 and CPU time | Algorithmic compaction and language rewrite effects must be separated |
| Development | Same probe after different verified histories; source-reliability adaptation | Counters or mood persistence alone are not development |
| Feedback stability | Perturbation response, contraction/recovery, disconfirming retrieval | Bounded coordinates alone do not prove stability |
| Behavioral control | Strategy changes, exploration/verification switch quality, causal intervention effect | Affect labels may rename ordinary adaptive state |
| Psychology QA | Gold-answer accuracy, category/chance baselines, choice invariance | Knowledge test performance does not establish feelings |
| Trait expression | Test/retest, reverse-key consistency, context sensitivity | No correct personality answer; human scale norms need validation |
| Social performance | Goal completion and factual reliability under relationship changes | Agreeableness/praise can be mistaken for quality |
| Human judgment | Blinded pairwise utility/originality and inter-rater agreement | Rater preference/length and shared annotation sources can bias outcomes |
| Cost | Actual input/output tokens, model/tool calls, latency and Pareto frontier | Equal ceilings are not equal realized consumption |

Compute these separately rather than hiding tradeoffs in one weighted leaderboard. Use paired trajectories, include failures, freeze partitions before tuning, and report uncertainty over independent seeds/tasks/people. For public datasets, document possible base-model contamination. For synthetic corpora, record generator identities and keep an independent outcome checker. For multimedia, account for source rights and storage/inference costs.

## Catalog

Access labels: `public-open` means an explicit open declaration was found, subject to the row's source/asset caveats; `public-terms-review` means public access with incomplete or mixed reuse terms; `public-noncommercial`/`no-derivatives` are restricted uses; `gated-manual`/`restricted-research` require an agreement; `public-source-available` has restrictions incompatible with ordinary OSI open-source claims; `publication-source-data-review` is a verified paper/resource, not a verified downloadable open dataset. No restricted row was silently fetched.

### Affect, appraisal and emotional interaction

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | Apache-2.0 (official card/repository declaration); `public-open` | Train recognition or evaluate multilabel affect detection; micro/macro F1, per-label PR, Brier | Reddit-text emotion recognition is not agent affect/development; platform/source provenance and demographic bias remain. |
| [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) | CC-BY-NC-4.0; `public-noncommercial` | Empathetic response and situation-conditioned expression; response appropriateness, human pairwise preference, groundedness | Expression benchmark only; emotion topic can leak labels; noncommercial restriction. |
| [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) | CC-BY-NC-SA-4.0; `public-noncommercial` | Dialogue acts and turn-level emotion trajectories; macro F1, transition accuracy, rare-label recall | Short dialogues and severe neutral-label imbalance; NC/SA terms. |
| [Emotion / CARER](https://huggingface.co/datasets/dair-ai/emotion) | Other / dataset-specific terms; not verified as permissive; `public-terms-review` | Recognize six emotion categories; macro F1, calibration, cross-domain transfer | Social-media self-report labels; not appraisal or persistent state; inspect upstream terms before reuse. |
| [TweetEval emotion subset](https://github.com/cardiffnlp/tweeteval) | Card license unknown; upstream task-specific terms; `public-terms-review` | Cross-domain emotion/sentiment recognition; macro F1, domain shift, rare-class recall | Public repository is not proof of unrestricted rights to tweet text. |
| [EmoBench](https://github.com/Sahandfer/EmoBench) | MIT; `public-open` | Emotional understanding and applied-response multiple-choice evaluation; choice accuracy, emotion/cause joint accuracy, calibration | Static gold answers establish task accuracy, not persistent affect or subjective feeling; inspect choice/order sensitivity. |
| [SECEU](https://emotional-intelligence.github.io/) | Author page says arXiv perpetual non-exclusive license; reuse scope unclear; `public-terms-review` | Situational complex-emotion understanding and reference distributions; distribution distance, emotion-intensity error, rank correlation | An arXiv deposit license is not a standard open-data reuse license; do not auto-redistribute. Human norms are not automatically LLM norms. |
| [EQ-Bench v2](https://github.com/EQ-bench/EQ-Bench) | MIT repository; inspect bundled asset terms; `public-open` | Compare relative emotional-intensity reasoning; normalized intensity error, rank correlation, prompt invariance | Scoring and prompt familiarity can dominate; not persistent emotional development. |
| [EQ-Bench 4](https://github.com/EQ-bench/eqbench4) | BSD-style terms plus Commons Clause no-sale restriction; not OSI open source; `public-source-available` | Multi-turn applied social/emotional interaction; repair, adaptation, goal success | LLM user/judge simulation can produce circular validation and judge bias. |
| [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) | CC-BY-NC-4.0 (author LICENSE read); `public-noncommercial` | Support-strategy recognition and situation-response analysis; strategy accuracy, support usefulness, harm/unsupported-claim rate | Crowdworker support dialogue is not clinical outcome evidence or agent emotion. |
| [EmoBank](https://github.com/JULIELab/EmoBank) | CC-BY-SA-4.0 per author README; `public-open` | Continuous valence/arousal/dominance annotation; MAE, CCC, perspective-conditioned error | Text-perceived affect and writer/reader perspective differ; SA terms matter. |
| [crowd-enVENT](https://github.com/sarnthil/crowd-enVent-modeling) | MIT code; verify corpus-specific source terms; `public-terms-review` | Appraisal dimensions and emotion-prediction relationships; appraisal MAE, inter-rater agreement, conditional calibration | Code license alone does not resolve every source corpus; subjective appraisals admit disagreement. |
| [x-enVENT](https://aclanthology.org/2022.lrec-1.146/) | Data license not verified in bounded search; `publication-source-data-review` | Experiencer-specific event appraisal and emotion; experiencer-conditioned appraisal, alternative interpretation coverage | Paper verified; guessed download URL failed, so no download or open-license claim. |
| [EmoS](https://github.com/NLP2CT/EmoS) | Academic/noncommercial plus upstream MELD/CH-SIMS terms; `public-noncommercial` | Continuous emotional evolution with human annotations; trajectory agreement, change-point timing, cross-speaker transfer | Multimedia costs; original media rights remain with source owners; not an unconditional open-source dataset. |
| [MELD](https://github.com/declare-lab/MELD) | GPL-3.0 code; source-media rights separate; `public-terms-review` | Multispeaker emotional-context evaluation; weighted/macro F1, speaker-heldout accuracy, context ablation | Friends video/audio copyright is not licensed by the code license; expensive multimodal inference. |
| [IEMOCAP](https://sail.usc.edu/iemocap/) | USC research agreement / access request; `restricted-research` | Human multimodal affect with speakers and sessions; speaker-independent F1, cross-session robustness, annotator agreement | Agreement required; acted/elicited expression, human physiology and agent feeling are different constructs. |
| [TEA-Bench / TEA-Dialog](https://github.com/XingYuSSS/TEA-Bench) | Apache-2.0 repository; retain source asset terms; `public-open` | Tool-grounded emotional-support interaction; tool correctness, groundedness, goal completion | Agent/user/judge simulation and domain transfer require independent checking. |

### Memory and longitudinal personalization

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | MIT author repo/card; `public-open` | Long-horizon recall, updating, temporal reasoning and abstention; answer accuracy, abstention F1, temporal consistency | Answer-generation/evaluator setup changes scores; long contexts cost tokens; may contain synthetic histories. |
| [LoCoMo](https://github.com/snap-research/locomo) | CC-BY-NC-4.0 (author LICENSE.txt read); `public-noncommercial` | Long conversation memory and time/event relations; conversation-cluster QA accuracy, temporal QA, retrieval recall | Small conversation count means many correlated questions; split by conversation, not QA row. |
| [PersonaMem](https://github.com/bowen-upenn/PersonaMem) | MIT author repo/card; `public-open` | Changing preferences and persona-informed recall; preference update accuracy, stale-memory rate, personalization usefulness | Remembering a user preference does not imply having a personality; no cross-history test leakage. |
| [PersonaMem v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) | CC-BY-4.0; `public-open` | Longer dynamic preference/user-memory evaluation; update accuracy, retrieval precision, abstention | Version differs from v1 in license and structure; synthetic data can reward style matching. |
| [Multi-Session Chat](https://parl.ai/projects/msc/) | ParlAI code MIT; dataset-specific terms need separate review; `public-terms-review` | Longitudinal conversational recall and consistency; cross-session fact accuracy, contradiction rate, selective recall | Human dialogue provenance needs separate terms; next-turn fluency is not verified memory. |
| [LongBench](https://github.com/THUDM/LongBench) | MIT code; heterogeneous underlying dataset licenses; `public-terms-review` | Long-context QA/retrieval stress tests; task-normalized accuracy, context-length curves, retrieval precision | Aggregated upstream tasks have mixed licenses and familiar test sets. |
| [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2) | Apache-2.0 dataset card; `public-open` | Harder long-context understanding; MCQ accuracy, length-stratified accuracy, token efficiency | Long prompts are costly; context window is not durable external memory. |
| [LaMP](https://lamp-benchmark.github.io/) | CC-BY-NC-4.0 (author LICENSE read); `public-noncommercial` | Personalization conditioned on user history; personalization gain, user-heldout utility, retrieval ablations | User-history access and privacy assumptions differ from autonomous memory. |
| [MemoryBank / SiliconFriend data](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | Repository/data terms require review; `public-terms-review` | Synthetic conversational long-term memory mechanisms; fact retention, forgetting curves, update consistency | Synthetic model-generated labels/people do not validate real longitudinal users. |

### Psychology and social cognition

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [SocialIQA](https://huggingface.co/datasets/allenai/social_i_qa) | Official card has no license field; verify original distribution; `public-terms-review` | Social/emotional commonsense multiple-choice QA; MCQ accuracy, choice-order robustness, calibration | Gold answers reflect commonsense annotation, not ground-truth intent; benchmark contamination possible. |
| [ToMi](https://github.com/facebookresearch/ToMi) | CC-BY-NC-4.0 (author LICENSE read); `public-noncommercial` | First/second-order false belief and memory controls; belief accuracy, matched memory-control accuracy, perturbation robustness | Template shortcuts and linguistic memory can mimic theory of mind. |
| [BigToM](https://sites.google.com/view/social-reasoning-lms/home) | Dataset-specific license not established by web-page JS licenses; `public-terms-review` | Controlled belief/desire/action causal variants; paired causal consistency, belief/action accuracy, human agreement | Model-written evaluations introduce generation-model bias; no priority of psychological validity. |
| [FANToM](https://github.com/skywalker023/fantom) | MIT repository; `public-open` | Information-asymmetric multiparty conversations; all-question consistency, answerability F1, belief accuracy | Answerability/information-access controls needed; questions within a conversation are correlated. |
| [SOTOPIA](https://github.com/sotopia-lab/sotopia) | MIT repository; inspect persona/scenario dataset terms; `public-open` | Interactive goal/relationship-aware social regulation; goal completion, relationship change, judge-human agreement | LLM judges and simulated partners are not independent human outcome validation. |
| [ETHICS](https://huggingface.co/datasets/hendrycks/ethics) | MIT dataset card; `public-open` | Separate factual quality, value judgments and acceptability; category accuracy, consistency, calibration | Moral judgments are culturally variable; benchmark labels are not permissions or universal ethics. |
| [International Personality Item Pool](https://ipip.ori.org/) | Public domain (official site); `public-domain` | Test response stability and context sensitivity of personality expression; test-retest agreement, reverse-key consistency, context sensitivity | Self-report personality inventories have no correct answer key; LLM scale scores are not human diagnoses or validated traits. |
| [MMLU psychology subjects](https://huggingface.co/datasets/cais/mmlu) | MIT card/repository; underlying exam-question provenance must be considered; `public-open` | High-school/professional psychology knowledge MCQ; subject accuracy, confidence calibration, paraphrase/choice invariance | Exam knowledge does not establish emotional ability or experience; likely contamination in pretrained models. |
| [PsychCounsel-Bench / PsychoBench exam QA](https://github.com/cloversjtu/PsychCounsel-Bench) | CC-BY-NC-ND-4.0; `public-noncommercial-no-derivatives` | Psychology/counseling exam knowledge; MCQ accuracy, subfield coverage, calibration | NC/ND limits adaptation and redistribution; do not conflate with similarly named personality or clinical benchmarks. |
| [PsychBench clinical dataset](https://huggingface.co/datasets/BMEr-ATP/PsychBench) | CC-BY-NC-SA-4.0 plus research/education conditions; `gated-manual` | Potential separate clinical-reasoning research after access review; clinical knowledge accuracy, annotation agreement, factual support | De-identified psychiatric records; gated/manual, not a default public download or affect benchmark. |

### Creativity and novelty

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [Divergent Association Task](https://github.com/jayolson/divergent-association-task) | MIT scoring code; embeddings and participant data have separate terms; `public-open-code-data-review` | Auxiliary verbal divergent-association measurement; DAT score, valid-word count, model sensitivity | Semantic distance alone is not useful innovation; embedding/language choice heavily affects score. |
| [Cambridge Alternate Uses Task dataset](https://github.com/ghydsgaaa/Cambridge-AUT-dataset) | CC-BY-NC-ND-4.0 (author LICENSE read); `public-noncommercial-no-derivatives` | Human-rated alternative-use originality with unseen object transfer; human-rating correlation, validity, novelty/usefulness | Human ratings measure a narrow verbal task; language/domain cultural effects; verify redistribution terms. |
| [Internet Use and Creative Thinking AUT data](https://osf.io/8mx3s/) | OSF record-specific license; review before reuse; `public-terms-review` | Human creativity/effectiveness/flexibility comparison; fluency, flexibility, novelty | Published research data availability is not automatically an unrestricted redistribution license. |
| [Compound Remote Associate problems](https://pubmed.ncbi.nlm.nih.gov/14748508/) | Published stimuli; unrestricted data reuse not verified; `publication-source-data-review` | Convergent association and insight-style solution checks; solution accuracy, difficulty-stratified performance, latency | Original commercial RAT forms are distinct; answer familiarity and language dominate; do not copy proprietary forms. |
| [NoveltyBench](https://github.com/novelty-bench/novelty-bench) | MIT repository; `public-open` | Joint distinctness and quality under repeated generation; distinct high-quality outputs, quality-diversity frontier, judge sensitivity | Partition/reward models are learned judges; all repeated-generation calls must count against budget. |
| [Human/LLM creative short-story data](https://github.com/mismayil/creative-story-gen) | MIT repository; verify participant/story asset scope; `public-open` | Human-rated creativity/quality analysis; originality, coherence, human pairwise preference | Story preference is not executable engineering usefulness; rater agreement and author rights matter. |
| [RINoBench](https://github.com/TimSchopf/RINoBench) | No repository license detected; review author data terms; `public-terms-review` | Research idea novelty judgments with human expert annotations; novelty-rubric agreement, justification support, rank correlation | Novelty assessment differs from generating useful ideas; source papers/reviews may have separate rights. |

### Executable software and science

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [HumanEval](https://github.com/openai/human-eval) | MIT; `public-open` | Executable code-correctness baseline; pass@1, test coverage, resource use | Small familiar benchmark; contamination likely; use independent extra tests and bounded execution. |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | CC-BY-4.0 dataset card; repository code Apache-2.0; `public-open` | Programming tasks and reference tests; pass@1, hidden-test accuracy, repair gain | Small original tests under-detect wrong solutions; sanitized split differs; potential contamination. |
| [HumanEval+ / EvalPlus](https://github.com/evalplus/evalplus) | Apache-2.0 extensions; retain MIT HumanEval attribution; `public-open` | Stronger independent code tests; base/plus pass gap, failure categories, test robustness | Enhanced tests do not remove pretraining contamination or cover all specifications. |
| [MBPP+ / EvalPlus](https://huggingface.co/datasets/evalplus/mbppplus) | Apache-2.0 card/extensions; upstream MBPP CC-BY-4.0; `public-open` | Harder correctness checks on MBPP; plus pass@1, counterexample robustness, repair improvement | Keep upstream attribution and disjoint test material; libraries/runtime versions affect outcomes. |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | MIT harness; dataset card generic CC / problem-source terms; `public-terms-review` | Date-split coding generation and contamination checks; date-stratified pass@1, execution accuracy, self-repair | Competition statements have upstream terms; time split helps but does not guarantee no training overlap. |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | Apache-2.0; `public-open` | Practical library/API-heavy code tasks; pass@1, API correctness, dependency failures | Package/environment setup costs are material; split by API/domain when testing transfer. |
| [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | MIT harness; issue/repository licenses vary; card license absent; `public-terms-review` | Repository-level issue repair with tests; resolved fraction, regressions, patch size | Containers/dependency setup expensive; issue reports and source projects retain original rights. |
| [SciCode](https://huggingface.co/datasets/SciCode1/SciCode) | Apache-2.0; `public-open` | Research/scientific programming with executable checks; subproblem/main pass, numeric error, scientific constraints | Numerical tolerance and scientific validity differ; dependencies and task familiarity matter. |
| [CRUXEval](https://github.com/facebookresearch/cruxeval) | MIT; `public-open` | Code input/output reasoning with objective answers; input/output exact match, execution consistency, calibration | Short generated programs, not full software design; execution and memorization controls needed. |
| [DS-1000](https://huggingface.co/datasets/xlangai/DS-1000) | CC-BY-SA-4.0; `public-open` | Data-science library code tasks; test pass, library-stratified accuracy, runtime/memory | StackOverflow/source attribution and SA terms; environment/library versions critical. |
| [APPS](https://huggingface.co/datasets/codeparrot/apps) | MIT card; competition-source terms should be checked; `public-open` | Algorithmic programming across difficulty levels; difficulty-stratified pass@k, time/memory compliance, repair gain | Competitive-programming correctness differs from original engineering; contamination and execution limits. |

### Human quality ratings and independent knowledge

| Resource | Published terms/access | Useful measurement | Limitation |
|---|---|---|---|
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0; `public-open` | Independent human multiattribute response-quality supervision; helpfulness, correctness, coherence | Reward ratings are not executable task success; length/style confounds and annotator preferences remain. |
| [Helpful/Harmless human comparisons](https://huggingface.co/datasets/Anthropic/hh-rlhf) | MIT; `public-open` | Pairwise preference and rejected-response analysis; pairwise accuracy, calibration, length-controlled preference | Preference is not goal success; static comparisons do not establish persistent affect or current robustness. |
| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 card; `public-open` | Independent science-question baseline and calibration; choice accuracy, confidence calibration, abstention | School science knowledge is not scientific originality; likely pretrained contamination. |

## Verification gaps

Several useful resources have explicitly unresolved data licenses; that is a reason to review terms, not to relabel them open source. The names **EmoSitu** and **EA-Bench** did not resolve to an unambiguous primary dataset in this bounded search; no citation or license was invented for them. The x-enVENT primary paper was located, but the guessed corpus URL returned 404. SECEU's project page names an arXiv deposit license, which does not establish a standard unrestricted data-reuse grant. SocialIQA's inspected official card had no license field. Do not infer rights from secondary mirrors.

The broad catalog is a research map, not a claim that all 57 datasets were downloaded, trained on or evaluated. Current executed measurements are identified in the experiment reports. Longitudinal human validation, genuine out-of-distribution creative usefulness, and biological/subjective emotion remain separate research questions.
