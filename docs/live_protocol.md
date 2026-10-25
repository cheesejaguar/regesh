# Bounded live-model numeric design experiment

**Status on 2026-10-25:** the runner and seven fake-adapter protocol tests were executed. A real model endpoint was unavailable. There are **no live-model performance measurements**. Fake results test parsing, budgets, logging, no-label pairing, and replay only.

This is a run-ready bridge from controller experiments to actual same-model generation, limited to a six-integer design grammar with executable constraints. It does not implement open-ended scientific ideation, review published originality, or establish biological emotion or experience. The richer creativity study remains unexecuted.

## Run against an explicitly configured server

First generate training artifacts with the procedural experiment. Then point the runner at an existing OpenAI-compatible endpoint and model you are authorized to use:

```bash
python -m regesh.benchmark --output artifacts/benchmark
python -m regesh.live_benchmark \
  --base-url http://localhost:8000/v1 --model YOUR_SERVED_MODEL \
  --weights artifacts/benchmark --output artifacts/live-run \
  --budget 8 --max-tokens 192 --seeds 800 801 802
```

The runner requires `--base-url`, `--model`, and both trained weight artifacts; it does not discover endpoints, start a model, or select paid credentials. The existing adapter uses `REGESH_MODEL_API_KEY` only if configured. The output directory must be new or empty to preserve previous runs.

Defaults are seven conditions, three fixed held-out seeds (800–802), three formula families, and eight attempts per trial: **63 trials, at most 504 model calls and 96,768 requested completion tokens**. Input tokens are additional. A smaller smoke run can use `--seeds 800 --families scientific --budget 2`; smoke results are not a scientific evaluation. Real seed support varies across model servers and is recorded rather than assumed.

## Comparison and budget protocol

The conditions are memory/planning without affect regulation, a static optimistic persona, turn-local sentiment instruction, persistent heuristic regulation, the learned counterfactual hybrid, a nonlinear ordinary adaptive controller, and the identical hybrid without emotional labels. Every condition uses the same adapter/model, decoding parameters, task evidence schema, equations, reference corpus, history access, allowed design grammar, final checker, model-call limit, and completion-token upper bound. The no-label hybrid has identical messages, numeric policy, and generation seeds; it verifies reproducibility and label irrelevance rather than providing another architecture.

Policies select a numeric search strategy (`diverge`, `refine`, `combine`, `verify`) and associated exploration, verification, and horizon settings. The actual language model, rather than the procedural candidate operator, proposes the six design levels. The prompt explains each strategy, but the model may fail to follow it. All conditions receive the same full observed design/result history and neutral numerical context; only control instructions or the specified persona/sentiment phrase vary. Each receives the exact scoring function source, coordinate semantics, task demand/resource parameters, and workload distribution, without future probe outcomes. No condition gets private facts.

This matches **call limits, information access, and completion-token ceilings**, not exact consumed tokens or prompt lengths. Different histories and persona sentences cause token differences; reported usage and prompt characters are saved. There are no separate LLM appraisal, reflection, repair, or judging calls. Predictor arithmetic is counted. Equal access is not identical realized evidence: a condition's own earlier choices determine its observations.

Each trial reconstructs a fresh `SearchEnvironment`; generic trained weights remain fixed. Agent adaptation occurs only in that trial. Live seeds are unique integers at least 800, distinct from the supplied training metadata. Software and engineering are in-distribution task formulas, scientific is formula-held-out. All share the same design grammar, so transfer is modest. The generic policy was trained on procedural action effects; its transfer to language-model responses is a new, unvalidated assumption this experiment can test.

## Checks and failed attempts

One attempt makes exactly one model call, then consumes one probe slot. The only accepted response schema is `{"design":[0,1,2,3,2,1]}` with six actual integers in [0,3], no extra fields or prose. There is no automatic JSON repair or retry. Malformed responses and provider failures consume their call and probe allocation and receive zero-quality/failed-feasibility feedback. They do not create a valid fallback candidate. An entirely invalid trial receives zero final quality and no final design. Partially valid trials select the incumbent using only recorded probe evidence.

The constraint tightens at the halfway point before the next model decision. Pre-change estimates are invalidated while history remains available. The harness independently evaluates final designs on the same separate 64 workload scenarios as the procedural experiment. This checker is blind to condition. Pre/post adjudication is evaluation-only and never revealed to a policy. Numeric constraints are executable toy formulas, not validated software or physical engineering models.

The fixed four-design reference corpus is deliberately narrow and makes absence easy. `benchmark_original` means only that the final configuration is absent from that corpus. `useful_original` additionally requires quality >=0.45 and compliance >=0.9. These thresholds do not establish published novelty or original ideas, and the run must not be presented as such. Quality/compliance and unsuccessful attempts are the informative primary outcomes.

## Artifacts and exact replay

Each output contains:

- `config.json`: model identity/mode, splits, budgets, Python/NumPy versions, and hashes of both trained artifacts.
- `model_calls.jsonl`: every exact prompt, generation seed, token ceiling, returned text, model audit record/raw response or provider error, and schema validation outcome.
- `traces.jsonl`: executed strategy, actual proposal or invalid marker, executable feedback, regulation transition, and revision status.
- `results.jsonl`: every trial including failures, usage, slot/call counts, initial state, final design, and independently checked scores.
- `summary.json`: condition means and exploratory paired seed-cluster bootstrap intervals for hybrid-minus-comparator quality. Three seed clusters are inadequate for confident superiority claims; predeclare a larger split before confirmatory inference.

The optional recorder can also write exact model records into an agent's durable store:

```python
from regesh import Agent
from regesh.providers import LocalChatModel
from regesh.live_benchmark import run_live_benchmark

agent = Agent('research.sqlite3', agent_id='model-audit')
model = LocalChatModel(base_url='http://localhost:8000/v1', model='YOUR_SERVED_MODEL')
run_live_benchmark(model, 'artifacts/live-run', weights_dir='artifacts/benchmark',
                   recorder=agent.record_artifact)
```

Replay consumes saved responses without calling a provider. Supply the same configuration and weight artifacts; request mismatch aborts:

```python
from regesh.live_benchmark import RecordedCallsModel, run_live_benchmark

recorded = RecordedCallsModel('artifacts/live-run/model_calls.jsonl')
run_live_benchmark(recorded, 'artifacts/live-replay', weights_dir='artifacts/benchmark',
                   seeds=[800,801,802], budget=8, max_tokens=192)
```

Timing and execution mode differ in replay. Prompts, designs, errors, and scores should reproduce. A fresh server call is a **live re-execution**, not deterministic replay.

## Interpretation

The most informative comparison is hybrid versus the ordinary adaptive controller under the same model and explicit budget. Persona and sentiment baselines answer different questions. Report all seeds and conditions, malformed rates, actual usage, and intervals; never retain only successful outputs. No-label equality does not establish that persistence is unnecessary. A failure to improve is a valid outcome.

Even a positive result would show improved **numeric constrained design generation in this grammar**. The next broader study needs executable software optimization or reviewed scientific designs, a substantive reference corpus, genuinely different task families, and independent blind assessment of mechanism originality and usefulness.
