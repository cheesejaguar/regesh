# Regesh

A local research platform for persistent functional affect: versioned evidence and appraisal, durable agent state, operational resources, developing expectations, affect-modulated cognitive control, causal interventions, and provider-free replay.

**Current evidence:** compact journaling reduces measured retained storage by about 97%, and snapshot lookup reduces isolated peak RAM by 89.5% at 250 events. Rust works as an optional backend but showed no dependable CPU/RAM gain. A completed real-model software pilot also found **no demonstrated affect advantage**: hidden pass rates 29.17% vs 28.24%, paired difference +0.93 points, 95% interval [−4.17, 6.94]. Neither controller produced a fully correct solution. See the [extended report](docs/extended_report.md) for real-model QA, human-label training, all failures, and multiple quality measures. Biological fidelity and subjective experience remain unestablished.

**Published-paper experiments:** [three core reproductions and a fourth scoped
replication](docs/paper_reproductions.md) are complete. They include 482 million
homeostatic transitions, 190 million bandit choices, original Chain-of-Emotion
data reanalysis and 64 new EmotionPrompt model calls. Results are mixed. Fixed
comparison exactly matches ordinary optimistic learning, and the prompt study
exposes a format confound. These are measured limitations, not affect superiority.

## Run locally

Python 3.11+ and NumPy; CPU only. No API key is needed for offline work.

```bash
python -m pip install -e .
python -m regesh demo --db demo.sqlite
python -m regesh inspect --db demo.sqlite
# Open http://127.0.0.1:8765
python -m unittest discover -s tests -v
python -m regesh core-eval --output results/core-new
python -m regesh benchmark --output results/benchmark-new
```

The full default benchmark trains on 6,144 observed transitions and runs 88 development plus 660 test trials, 32 objective probes per trial, across 11 conditions. Every condition shares information, action operators, model-prediction counts and probe budget. Actual timings and all raw evidence are retained. Use a new output directory to preserve a run. Distributed raw JSONL archives may be gzip-compressed; `gzip -dk FILE.jsonl.gz` restores them. Fresh runs produce plain JSONL.

## SDK

```python
from regesh import Agent, Event

agent = Agent('agent.sqlite', agent_id='researcher')
event = Event('test-1', 'unit-tests', 'checker', 'proposal', 'software',
              '2026-10-25T00:00:00Z',
              data={'goal_delta': {'quality': -.5}, 'control': .7})
agent.observe(event)
controls = agent.actuation_context()  # exploration, verification, retrieval, horizon
agent.act('verify')                  # allocation consumes operational resources
agent.record_artifact('tool_result', {'passed': False})
agent.record_outcome(event.id, -.5, provenance='independent executable checker')
snapshot = agent.snapshot()
child = agent.fork('comparison')
replayed = agent.replay('recomputed') # recorded appraisals; checks every poststate
assert agent.get_state() == replayed.get_state()
child.close(); replayed.close(); agent.close()
```

Observations and outcomes must come from appropriately trusted sources. Model-generated interpretations do not establish facts or grant tool permissions. Event identities and outcome identities are idempotent. Recall can reactivate state without counting as fresh evidence.

`observe`, `appraise`, `advance_time`, `get_state`, `actuation_context`, `act`, `record_outcome`, `retrieve`, `reappraise`, `snapshot`, `restore`, `fork`, `reset`, `intervene`, and `replay` form the SDK. See [architecture and time semantics](docs/architecture.md), [CLI/model/inspector interfaces](docs/interfaces.md), and [existing-agent example](examples/agent_loop.py).

## Learned mechanism and model adapter

```python
from regesh.learning import RidgePredictor
agent = Agent('agent.sqlite', agent_id='learned')
agent.set_regulator(RidgePredictor.load('results/benchmark/counterfactual_weights.json'))
agent.set_task_context({'quality': .4, 'uncertainty': .6})
print(agent.actuation_context())      # predicted alternative consequences and scores
agent.act('auto')
agent.close()
```

The saved model is trained in the toy simulator. Transfer to arbitrary host agents is unvalidated. Shared weights and an individual's adaptive history are saved separately.

`regesh.providers.LocalChatModel` supports an explicitly configured OpenAI-compatible server, including vLLM. It preserves exact requests/responses, seeds, timing and returned usage. Neither a GPU, hosted model nor credential is assumed. `FixtureModel` works offline. A same-model numeric-design comparison is runnable when a server is available:

```bash
python -m regesh.live_benchmark \
  --base-url http://localhost:8000/v1 --model YOUR_SERVED_MODEL \
  --weights results/benchmark --output results/live-new \
  --seeds 800 801 802 --budget 8 --max-tokens 192
```

Default ceiling: 504 model calls and 96,768 completion tokens; input tokens are additional. See [live protocol and recorded-response replay](docs/live_protocol.md). The initial study had no configured endpoint. The extension provisioned an open-weight CPU model and completed the software and psychology pilots; their measured results are separate from this original numeric-design runner and its fixtures.

## Compact memory, datasets and expanded experiments

Compact Python storage is the default. The optional Rust kernel uses the same
lossless representation; it is an evaluated alternative, not a blanket speed claim:

```bash
cargo test --locked --manifest-path native/Cargo.toml
cargo build --release --locked --manifest-path native/Cargo.toml
REGESH_MEMORY_BACKEND=rust python -m regesh demo --db rust-demo.sqlite
```

See [memory profiles and remaining scaling limits](docs/rust_memory.md).
`agent.iter_trace()` and default replay stream history; `trace()` explicitly
materializes it. Original numerical behavior is unchanged.

The [dataset catalog](docs/datasets.md) covers 57 resources with license/access
checks. Four have pinned verified downloads. The [psychology suite](docs/psychometrics.md)
uses external answer keys, response-order/persona checks and calibration/coverage;
it does not infer clinical traits or feelings. Full EmoBench conversion and a
small supervised human-label reference are runnable:

```bash
python scripts/fetch_datasets.py --cache .cache/regesh-data
python scripts/prepare_emobench.py --cache .cache/regesh-data --output .cache/emobench-all
python -m regesh.dataset_baseline --cache .cache/regesh-data --output results/recognition-new
```

To reproduce local real-model inference, use Python 3.12+ for the prebuilt helper
(or its source-build option), Linux x64 and about 1.12 GB model download:

```bash
python scripts/prepare_local_model.py --cache .cache/regesh-model --prebuilt-server
.cache/regesh-model/llama-b11878/llama-server \
  --model .cache/regesh-model/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 --alias regesh-qwen-coder \
  --ctx-size 8192 --parallel 1 --threads 4 --threads-batch 4 --n-gpu-layers 0
# In a second terminal:
python -m regesh.software_eval --base-url http://127.0.0.1:8080/v1 \
  --model regesh-qwen-coder --config configs/software_cpu_pilot.json \
  --weights results/benchmark --output results/software-new --timeout 300
python -m regesh.psych_eval --items datasets/emobench_pilot.jsonl \
  --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder \
  --permutation --persona --max-tokens 128 --timeout 300 --output results/psych-new
```

The completed CPU pilot used 48 software and 36 QA calls; the larger software
configuration remains available. Exact limits, raw costs, source/model hashes,
runtime corrections and seed nondeterminism are documented. [Recorded-response
replay](docs/software_replay.md) verifies every actor phase without a model and
separately rechecks the executable outcomes. [Expanded measurements](docs/measurement_framework.md)
and [frozen secondary analysis](docs/extended_analysis.md) keep correctness,
robustness, repair, cost, calibration, coverage and diversity proxies separate.

## Pair bonding with a user (opt-in)

The affect loop can keep a per-partner **bond**: attachment-style arousal
regulation with budgeted help-seeking (Hiolle et al. 2014), secure-base threat
buffering (Coan et al. 2006), an online-identified dyadic affect model with
bounded steering and repair (Gottman et al. 1999/2003), a mid-range style
coordination control (Danescu-Niculescu-Mizil et al. 2012; Jaffe et al. 2001)
and reciprocity-weighted intimacy (Laurenceau et al. 1998). Default agents are
unchanged; the committed demo digest is identical.

```python
agent = Agent('agent.sqlite', 'companion', config={'bonding': True, 'partner': 'user'})
bond = agent.actuation_context()['bond']   # request_support, security, dyad steering, coordination excess
agent.express('user', valence=bond['dyad']['expressed_valence'], text='...')
```

The bond never alters factual reliability, authority or permissions, and
requests are bounded by the partner's attention budget. Four new published-paper
experiments and a platform-level simulation accompany the mechanisms:

```bash
python -m regesh.reproductions.attachment_robot --output results-new/attachment-robot
python -m regesh.reproductions.marital_dynamics --output results-new/marital-dynamics
python -m regesh.reproductions.coordination_power --cache .cache/convokit/supreme --output results-new/coordination
python -m regesh.reproductions.llm_convergence --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder --output results-new/llm-convergence
python -m regesh.bonding.experiment --config configs/bonding_platform.json --output results-new/bonding-platform
python -m unittest tests.test_bonding tests.test_bonding_experiment -v
```

See the [bonding report](docs/bonding.md), the [novelty statement and
safeguards](docs/bonding_disclosure.md) and the per-paper reports under
`docs/reproductions/`. No result establishes benefit to users; see the report
for measured outcomes and their limits.

## Research and results

- [Published-paper reproductions, quantitative comparisons and run commands](docs/paper_reproductions.md)
- [Extended measured report with figures](docs/extended_report.md)
- [Initial research report and claim ledger](docs/research_report.md)
- [Prior-art matrix: 28 primary-source comparisons](docs/research_matrix.md)
- [Ten competing hypotheses and independent critic](docs/hypotheses.md)
- [Benchmark protocol](docs/benchmark_protocol.md) and [executed results](results/benchmark/report.md)
- [Core histories, interventions and feedback measurements](results/core/core_experiments.json)
- [Independent adversarial review](docs/adversarial_review.md), [experiment critique](docs/experiment_critique.md), and [pilot provenance](docs/exploratory_pilot.md)
- [Static inspector example](results/inspector.html) and [durable work log](docs/WORK_LOG.md)

Persistence, appraisal, multiple timescales, emotional memory, homeostatic control and creative search scheduling have substantial prior art. The possible research contribution is the precise learned operational-consequence coupling and its controlled evaluation; architectural originality is not established by naming the state emotional.
