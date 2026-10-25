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
