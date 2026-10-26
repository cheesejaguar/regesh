# Working checkpoint

2026-10-25. Repository cloned at b13ec00; only README/license, clean, no AGENTS.md or research map/PDF. Branch research/persistent-affect. CPU: 5 Intel Xeon virtual cores, ~17 GiB RAM, Python 3.12, NumPy, no CUDA/PyTorch, no configured model secret/endpoint. localhost ports 8000/11434 unavailable. No paid dependencies provisioned.

Parallel responsibilities: prior_art -> source matrix; hypotheses -> competing mechanisms and critic; benchmark -> learned mechanism + executable synthetic search; adversary -> independent tests and review. Main -> durable core, interfaces, integration, run experiments/report.

Core decision: explicit event evidence and appraisal versions; SQLite atomic per-agent poststate journal; operational resources; reference appraisal + target affect/mood/relationships/expectations; outcome-dependent learning; control affects retrieval, exploration, verification, learning rate, horizon. Learned benchmark mechanism predicts action consequences and projects resource/affect costs. Exact no-label clone and ordinary adaptive comparator required. Precedents invalidate broad novelty claims about affect persistence/memory/multiple timescales.

Budget: offline CPU experiments, initially 20 paired seeds per test scenario, fixed action-evaluation budgets, training/development/test separated, no live provider calls without configured endpoint. Raw outputs and full configurations retained. No human study. Controller/simulator results cannot establish live-LLM creativity.

Replay is provider-free recorded poststate playback with hash validation; it is deliberately distinguished from live re-execution or independent transition recomputation. Schema v1 experimental. Full-state journal trades speed/storage for auditability.

## Final checkpoint

Core and interfaces complete. Default replay was upgraded from initial poststate playback to full provider-free numerical recomputation using recorded appraisals, with every poststate checked; explicit playback remains available. All 64 tests pass (including subprocess persistence, adversarial histories, fake HTTP model calls, live-runner fixtures and transition-mismatch detection). Editable install, wheel packaging/default-config loading, CLI demo/inspector export/training/core evaluation, and example integration were executed.

Official corrected exploratory experiment: results/benchmark, 6144 training transitions, 88 development trials, 660 held-out test trials, 11 conditions, 20 paired seeds, 32 probes/trial. Hybrid quality .6130187938 vs structurally different nonlinear ordinary adaptive .6258466958; delta −.0128279020 CI[-.0370468,.0094043]. No demonstrated benefit. Hypothesis-specific ablations do not support internal projection/memory/temporal benefits. Label-free traces identical. Full rerun in /workspace/scratch/regesh-reproduction matches all 660 result records except latency, both weight files and complete action traces. results/reproduction.json records verification.

results/core contains finite feedback/development/intervention measurements and complete gzip history journals. results/benchmark stores exact source snapshots, manifests, weights, config, raw results, forecasts and executed actions. Large JSONL traces/training are losslessly gzip archived; fresh runs write plain JSONL. results/provenance.json hashes delivered code/config/tests/evidence. The official report is docs/research_report.md; the reviews preserve initial flaws and final resolutions. An initial pilot's raw files were overwritten before archival; log-only pilot metrics are disclosed in docs/exploratory_pilot.md and excluded from final evidence.

Live benchmark code is complete with explicit endpoint/model, finite call/token ceilings and exact recorded-response replay. No live model endpoint/credential was available; no real model called. No human/physiology evidence, no established architectural priority, no live-model creativity advantage. Next experiment: prospectively fixed same-base-model hybrid vs ordinary adaptive comparison on held-out executable software design/repair with matched total token/tool budget and substantive novelty reference corpus.

## Extended work: Rust, real model, datasets, measurement (2026-10-25)

User authorized Rust/performance exploration, lightweight memory, actual same-model software comparison, broad dataset research and psychology-style evaluation. Parallel agents delivered integrated Rust/Python compact delta storage, profiling, a57-resource license-aware catalog+four pinned dataset downloads, externally keyed EmoBench QA, independent software tasks/checker, and measurement review.

Memory default is now Python compact. Rust is optional because profiles did not establish a dependable latency/RSS win. At250events SQLite journal retention drops94.46->2.728MB(sparse) or187.70->6.406MB(clocked); snapshot latest-hash lookup reduces isolatedpeakRSS235460->24608KiB. Streaming replay/iter_trace avoids retaining all poststates. Dense clocked deltas and full current-state serialization remain limitations. Sources/manifests in results/memory-profile, memory-final-focus, snapshot-profile.

Actual model running locally: official Qwen2.5-Coder1.5BInstructQ4_K_M,1,117,320,768bytes,SHA256cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046; Apache2.0; llama.cpp b11878 prebuilt CPU releaseSHA256edadc40f684c3e44c7d1261eb72c993b98e66c3104555cbc43a0b261d773437f, sourcea55c16e4ee532aa36d55f5756f3ba28a21a2b62b. CPU endpoint http://127.0.0.1:8080/v1 modelaliasregesh-qwen-coder. Model/server binaries outsideGit under /workspace/scratch; preparation script is versioned. No paid API credentials. Four CPU equivalents quota; five visiblecores. Sourcebuild was stopped after verifying official prebuilt release.

Prospective full4seed software protocol committed9ec6fe6; unrelated throughput probe58tokens/15.66s motivated2seed CPU pilot BEFORE any held-out software generation, committedbc83c45. Executed config configs/software_cpu_pilot.json:6tasks×2seeds×2conditions=24trials,2candidatecalls/trial=48calls maximum24576completiontokens; one visible feedbacktool and2blindchecker calls/trial. Generic hybrid/direct weights frozen. Six authored taskfamilies, independent hidden/property/metamorphic/scaling checks. Protocol tests passed before live inference. QA uses12externally keyed EmoBench items×base/order/personavariants=36calls,<=4608completiontokens. QA measures emotional knowledge and answerreliability, not affect controller superiority or subjective emotion.

ACTIVE jobs at this checkpoint: software process(toolsession34631) writes results/software_real/{calls,traces,results}.jsonl and summary.json on completion; log /workspace/scratch/software-real.log. QA process(toolsession79883) writes results/psych_real/responses.jsonl and summary.json on completion; log /workspace/scratch/psych-real.log. They share one serial modelslot(server toolsession34936), so descriptive latency includes queueing. Do not duplicate runs or overwrite these directories. Exact plan/model/runtime info in results/runtime. Larger96call software configuration remains runnable. Next: await both completed studies, analyze all outcomes without tuning, append independent review, document results, verify native/fallback/tests and update PR1. Historical results/provenance.json belongs to initial commit2bf0e03; use a separate extended provenance manifest.

Runtime correction: the first QA client used a 60-second adapter deadline while sharing the serial CPU slot. It was stopped with five completed records (one timeout, four schema failures) and at most one in-flight call, all retained under results/psych_runtime_partial with an abort manifest. The full QA run restarted at tool session 53269 with a 300-second deadline; prompts, items, seeds, token cap and primary scoring are unchanged. Source gold accuracy was not inspected to choose this correction. New source-only secondary multi-axis analysis is being frozen before inspecting final correctness results.


## Completed extended checkpoint

All requested work is complete. The CPU service was stopped after measurement to release RAM; its pinned model cache, server archive and preparation helper remain available. No paid credentials or API were used.

- Compact memory retains all evidence and legacy logical hashes. At 250 events: retained journal 94.46→2.73 MB (append) and 187.70→6.41 MB (clocked); old snapshot materialization 235,460→24,608 KiB peak RSS. Default is Python compact; Rust is opt-in because CPU/RSS tradeoffs were mixed. Default replay now streams. The original core JSON measurements and trajectories are exactly unchanged (results/core_compact_recheck).
- Real Qwen coder experiment completed 24 trials / 48 calls, six task families / two seeds / two frozen controllers. Hybrid hidden pass fraction 0.291667, ordinary 0.282407; difference +0.009259, 95% task-cluster interval [-0.041667,0.069444]. No full-suite success. Same-suite revision worsens both. Actual actions differ in 8/12 second decisions. Only 7/12 identical initial prompts/seeds return identical text; fresh decode is not bitwise deterministic.
- Complete QA run: 36 attempts, zero provider errors, all strict-schema failures. Pre-frozen exact-fence/legal-letter diagnostic: 18 choices extracted, 11 correct overall; base questions 6/12 extracted, 4/12 correct. This is output-protocol failure, not zero emotional knowledge. Original and corrected macro-F1 summaries/scorer source are both retained. The aborted 60-second runtime pilot is preserved separately, with five completed attempts and at most one in flight.
- Record-only software replay verifies all 24 trajectories / 48 actor phases with zero model/network calls. Separate 48 executable rechecks all match stable outcomes. results/software_replay contains proof. All main fitted weights remain unchanged.
- Dataset catalog: 57 resources, 27 with permissive declarations plus one public-domain source; other terms/gaps are explicit. Four datasets / nine artifacts downloaded and SHA/size verified. Full EmoBench converter produces 400 English items (600 including causes over the same 400 scenarios) and regenerates the locked pilot byte-for-byte.
- Human-label GoEmotions reference: 39,013 fit / 4,397 development / 5,427 test. Micro-F1 0.414123 vs prior 0.304015; macro-F1 0.223103. Exact-set accuracy, Hamming loss and calibration worsen; five rarest labels F1=0. Learned arrays 458,976 bytes, compressed model 251,343 bytes. No main-controller changes. Independent stdlib counting reproduces primary metrics.
- Expanded measurements include correctness, boundary/generated/metamorphic/scaling robustness, same-suite repair/regression, confidence/coverage, proper scores, abstention/risk, cost, source proxies, source reliability and finite feedback/recovery. Undefined metrics remain missing. Four standalone SVG/PNG figure pairs are in results/figures.
- Final verification: 141 Python tests pass in BOTH backends, plus two Rust tests; wheel builds. Historical results/provenance.json remains tied to initial commit 2bf0e03. New code/evidence hashes are in results/extended_provenance.json. The main current report is docs/extended_report.md. Final next experiment: common checked initial-history fork, a stronger/appropriate model with baseline full-task success, and validated decoding/replay before a larger fixed comparison.

## Published-paper reproduction checkpoint

The user's subsequent request for at least three relevant paper experiments is
complete: two full-scale independent computational experiment implementations
(Keramati/Gutkin2014 and Dubey/Griffiths/Dayan2022), one original-data statistical
reproduction (Croissant etal2024), plus a scoped real-model EmotionPrompt
replication (Li etalICML2024). Primary sources, fidelity labels, direct numerical
comparisons, failed matches and commands are in `docs/paper_reproductions.md`.
Detailed durable notes are `docs/reproductions/WORK_LOG.md`.

Executed482million homeostatic transitions,190million bandit choices,42original
STEUscore rows,73completeLIWCrows,30×3originaluser ratings and64newmodelcalls.
Homeostatic maximum86vs61 and stationary fixed-comparison reward reversal remain
unreproduced. Sudden-change reward contrast370.82 closely matches370.11. Fixed
comparison exactly matches ordinary optimistic learning across40million paired
actions. STEUcounts24/31/35 reproduce. Questionnaire inference depends on analysis
assumptions. Promptaccuracy19/32→22/32 is inconclusive and format-sensitive;
exploratory arithmetic correctness is12/16both. No affect-specific creativity
advantage is established. The model service is stopped; no jobs remain.

All173Python tests, independent raw-result audits and compileall pass. New hashes
are in `results/papers/provenance.json`; all prior provenance/evidence/weights are
preserved. Draft PR1 now includes runnable papers, quantitative figures and the
independent review. It remains unmerged.

## Pair-bonding extension checkpoint (2026-10-25/26)

User request: expand the affect system to pairing/bonding with its user using
biomimicry and cognitive-science methodology, implement three high-impact
changes grounded in published papers, reproduce their findings, iterate.

- Mechanisms: `regesh/bonding/` (attachment, dyad, synchrony, platform bridge,
  experiment). Opt-in via `bonding: True`; default agents unchanged (demo digest
  identical); 221 tests pass.
- Reproductions: Hiolle 2014 (direction reproduced, absolute pace and literal
  adaptive rule not), Gottman 1999/2003 (steady-state structure clarified, the
  estimator imprecise at 150 points, damping claim 0/360), Echoes of Power
  (2 of 4 Supreme Court claims), Blevins 2026 (over-convergence not reproduced
  with the 1.5B model; bounded instruction a null).
- Platform iteration: first protocol showed prior-dominated security and
  cap-saturated steering; second protocol adopted the Beta evidence-count
  estimator (now default), a responsiveness floor on requests, and a reactant
  partner that separates identifier-based steering from fixed positivity.
- Safeguards: trust firewall, attention budget, responsiveness floor, no
  distress or guilt at disengagement; documented in `docs/bonding_disclosure.md`.
- Local model server used for 400 calls plus smoke tests; stopped after the
  study. DailyDialog (ParlAI mirror) and ConvoKit term files cached outside git.
- PR #3 on `research/pair-bonding`.

## Learned components checkpoint (2026-10-26)

User asked how heuristic versus learned the bond is, and to pursue a small
network paired with the LLM (slot 1: partner-state sensor) and a learned
partner model (slot 2) in parallel.

- Shared: `regesh/bonding/mlp.py` (NumPy MLP, finite-difference-checked) and
  `regesh/bonding/embeddings.py` (content-addressed cache of the generating
  model's mean-pooled embeddings; GoEmotions and DailyDialog subsets, ~53k texts,
  about 3 h on CPU). Server stopped afterwards.
- Slot 1: logistic head reaches micro-F1 .486 / macro-F1 .348 on the official
  GoEmotions test (hashed reference .414 / .223 at 100% rows; .370 / .141 like
  for like at 35%); DailyDialog 7-class accuracy .843 vs majority .817; zero-shot
  valence sign .894 on emotional turns but a positive bias on neutral turns;
  post hoc neutral gate selected on validation (tau .3, dead band .4) is the
  `annotate_event` default. Non-deployed MLP heads gzipped in results.
- Slot 2: windowed MLP over embeddings adds nothing to next-affect prediction
  (RMSE +.007 vs scalar features); its gain over the online identifier was cold
  start; the identifier now warm-starts from the pooled DailyDialog fit (RMSE
  .244 vs .275), diffuse start selectable; persistence predicts the next
  emotion class best.
- Full suite 266 tests pass; demo digest unchanged; audit covers nine manifests.

