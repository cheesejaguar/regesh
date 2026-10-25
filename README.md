# Regesh

A local research platform for persistent functional affect: versioned evidence and appraisal, durable agent state, operational resources, developing expectations, affect-modulated cognitive control, causal interventions, and provider-free replay.

**Current evidence:** the system passes persistence/development/control tests, but the exploratory simulator experiment does **not** demonstrate better useful solution generation than ordinary adaptive control. Hybrid quality 0.6130 vs adaptive 0.6258; paired difference −0.0128, 95% interval [−0.0370, 0.0094]. No live-model creativity, biological fidelity, or subjective experience has been established.

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

Default ceiling: 504 model calls and 96,768 completion tokens; input tokens are additional. See [live protocol and recorded-response replay](docs/live_protocol.md). Live evaluation was blocked here by the absence of a model endpoint/credential and GPU runtime. Fake HTTP/adapter tests validate the protocol only.

## Research and results

- [Research report and claim ledger](docs/research_report.md)
- [Prior-art matrix: 28 primary-source comparisons](docs/research_matrix.md)
- [Ten competing hypotheses and independent critic](docs/hypotheses.md)
- [Benchmark protocol](docs/benchmark_protocol.md) and [executed results](results/benchmark/report.md)
- [Core histories, interventions and feedback measurements](results/core/core_experiments.json)
- [Independent adversarial review](docs/adversarial_review.md), [experiment critique](docs/experiment_critique.md), and [pilot provenance](docs/exploratory_pilot.md)
- [Static inspector example](results/inspector.html) and [durable work log](docs/WORK_LOG.md)

Persistence, appraisal, multiple timescales, emotional memory, homeostatic control and creative search scheduling have substantial prior art. The possible research contribution is the precise learned operational-consequence coupling and its controlled evaluation; architectural originality is not established by naming the state emotional.
