# Architecture and semantics

## Implemented loop

`Event → validated appraisal → fast target state / mood → controls → resource-constrained cognitive allocation → observed outcome → expectations, source reliability and episodic memory`.

This is persistent regulation inside a cognitive loop whose optional LLM already has learned affect representations. Model weights and an individual agent's saved experience are distinct. There is no claim of an emotionless reasoning model plus a biologically literal emotional organ.

The core uses Python/SQLite and NumPy only for learned controllers/experiments. `schema.py`, `storage.py`, `appraisal.py`, `dynamics.py`, `agent.py`, `learning.py`, `providers.py`, `benchmark.py`, `evaluation.py`, and `inspector.py` separate evidence, persistence, interpretations, transitions, orchestration, learned models, language providers, simulators, evaluation and inspection. The SDK is the local API. The inspector also provides JSON endpoints. No network service is necessary for the SDK.

## Evidence and appraisal

Schema-v1 events include identity, source, actor, target, task, timezone-qualified timestamp, kind, confidence, data and causal IDs. Kind distinguishes occurrence, quotation, recall, prediction, imagination and reflection. A duplicate identity with identical evidence is idempotent; a changed payload is rejected. Nested mutable event data are revalidated on observation. There is one verified outcome per occurrence; corrected evidence should be a new event, followed by a versioned reappraisal, not a rewrite of historical evidence.

The reference appraiser computes weighted goal gain and relevance, prediction surprise, learned control, responsibility, reported approval, uncertainty and alternatives. `goal_delta` is an input observation supplied by a fixture or environment adapter; the appraiser does not independently prove it. The source must be trusted appropriately. Outcome quality is supplied by a checker; praise does not supply quality. Responsibility is supplied explicitly rather than guessed from language. Alternatives remain inspectable rather than being numerically inferred as true intent.

Optional `ModelAppraisal` proposes the same schema and records the exact request/response. Bounds and evidence IDs are checked; these checks do not validate the truth of the LLM's interpretation. Model observations cannot silently become measured outcomes. Permissions and tool authorization remain with the host agent. Control outputs grant no new permissions.

## State and timescales

Each event retains satisfaction, threat and curiosity components plus a target. Multiple positive/negative components and targets can coexist. The reference transition has confidence-weighted habituation and goal-loss-dependent threat:

```
intensity = confidence * (1 if occurrence else .15) / (1 + .3 * repetitions)
threat = max(0, -gain) * (1 - .65 * controllability) * (1 + .2 * uncertainty)
mood.valence = .88 * old_valence + .12 * (satisfaction - threat)
mood.arousal = .85 * old_arousal + .15 * (threat + curiosity)
```

These are transparent engineering equations, not fitted psychological laws. Fast components decay with simulation-time tau=3, mood tau=24, relationship affiliation tau=240, and disposition optimism tau=1000. Event transitions supply a second, discrete decay; event frequency therefore matters. Time constants are fixed reference parameters, not learned timescales. Repetition counts recover with tau=12. Factual reliability does not evaporate with elapsed time.

Relationships separately retain familiarity, affiliation, Beta factual reliability and authority. Familiarity and approval affect neither authority nor independently checked factual success. Authority is a reserved externally governed field, initially zero; it does not confer tool permissions. Outcomes with an independently checked `partner_correct` flag update factual reliability, which changes future uncertainty. Person identity can be held out at integration level; the current procedural benchmark has no simulated social partners.

Outcome learning changes future appraisal: task expectations and controllability update toward verified outcomes, and prediction errors alter slow precision/optimism statistics. Per-agent expectation learning is separate from generic ridge training. Shared trained weights remain fixed during held-out trials. Recall/reappraisal never increments outcome-learning counts.

## Operational embodiment

| Variable | Independent dynamics / consequence | Restoration or constraint |
|---|---|---|
| resources | Cognitive allocations consume .02–.12; remaining resources bound execution and planning horizon | `recover` restores .25; explicit elapsed time restores .025/unit; exhausted allocation rejected |
| unresolved commitments | `commit` creates and `resolve`/verified outcomes clear obligations; commitments shorten horizon | Hard capacity 64; resolve costs resources |
| operational reliability | EMA of verified quality; low reliability raises verification effort | Further verified successful work improves it; praise does not |
| epistemic uncertainty | Appraisal and outcome mismatch update it; increases verification and reduces exploration | New outcomes with low mismatch reduce it; allocating verification alone is not evidence |
| processing burden | Allocations accumulate burden, which shortens horizon | Explicit rest/review and elapsed time reduce it |

High load alone does not create threat. The profile models computational/organizational constraints, not hunger, pain or physiology. The benchmark has a separate simulated review-burden profile: review lowers burden, but burden does not disable an action. That benchmark's internal score penalty is not an externally measured resource limit; design-resource constraints are independently checked.

## Memory and reappraisal

Observations, outcomes and interpretation versions are separate. Causal IDs give a lightweight graph, not learned graph embeddings. Encoding salience uses gain/surprise. Retrieval mixes task relevance, salience and mood congruence; `evidence_only=True` provides a mood-independent path through occurrences. Research reactivation is optional and explicit; it produces a reduced affect transition without increasing familiarity, observations or factual learning. Retrieval counts and selection are logged. Reappraisal requires recorded new occurrence IDs, retains the old observation and versions the new interpretation. It updates affect at reflection intensity rather than fabricating a new occurrence.

The finite feedback experiment compares two initial moods under coupled and disconnected retrieval, then stops reactivation. Feedback preserves a separation; cessation reduces it. This is a measured limitation as well as a stability check. It does not prove stability for arbitrary histories or LLM loops. Full storage is retained; a learned consolidation/forgetting subsystem was not implemented.

## Cognitive control and learned component

Reference affect changes exploration, verification, retrieval ordering, outcome learning rate and horizon. Allocations really consume resources. A host agent must connect the numeric controls to its own candidate generation/tools; the example shows that integration. Setting a mood value is a causal intervention, not evidence about an agent's subjective experience.

The experimental component fits a four-head, action-conditioned ridge predictor on randomized simulator transitions. Inputs: 17 operational/history features with no task-family or person ID. Four action blocks × 18 features including bias × four heads = 288 parameters. Heads predict quality improvement, uncertainty reduction, burden increment and feasibility. One-step scores combine predicted task consequences with current-state-dependent internal penalties. Coefficients are declared in source; they were not optimized against test scores. No recurrent network or affect-name supervision is used: persistent state is explicit, and training targets come from executable transitions.

`Agent.set_regulator(predictor)` persists trained weights and their identifier separately from adaptation. `set_task_context()` journals named task measurements. Learned control affects actual retrieval and `act('auto')` allocations. Applying these simulator-trained weights to arbitrary real agents is unvalidated; the adapter reports that scope. The benchmark itself uses its own complete `SearchState` and simulator dynamics, not the SQLite reference dynamics. Therefore its measured search effects do not validate the reference appraiser's numerical equations.

Distributed **prediction heads** are learned; maps from selected strategy to control values are hand-authored. The B3 hypothesis of learned process-specific modulation receptors is **not fully implemented**. B2 is the selected implemented experimental mechanism. Other candidates remain hypotheses.

## Pair bonding (opt-in)

With `config={"bonding": True, "partner": "<actor>"}` the state gains `attachment`
(agent-level arousal, comfort trace, pending request) and `bonds` (per actor:
responsiveness, adapted comfort profile, attention budget, intimacy, disclosure
depths, an online dyadic model, coordination counts). Default agents have none
of these keys, and the committed demo snapshot digest is unchanged.

Equations are published ones, implemented as pure functions in `regesh/bonding/`:

```
comfort  = (comfort*tau_h + support)/(tau_h+1) if support>0 else beta_h*comfort      # Hiolle et al. 2014, Eq. 4
arousal  = (tau_ar*arousal + stim)/(tau_ar+1) if comfort<=.1 else arousal - alpha_ar*comfort   # Eq. 3
stim     = clip(goal_relevance*2*surprise + threat, 0, 1)                            # relevance-gated prediction error + threat
resp    += alpha_resp*(1-resp) if answered else -alpha_resp*(1-resp)                 # Eq. 5, per resolved request ("ema")
resp     = (answered+1)/(answered+unanswered+2)                                       # or the Beta evidence-count estimator ("beta")
beta_h   = .7 + (1-resp)*.3 ;  tau_h = 3 + (1-resp)*7                                # Eqs. 6-7
threat  *= 1 - gain*resp*familiarity*presence                                        # Coan et al. 2006 buffer
P_{t+1}  = b + r*P_t + s_neg*min(A_t,0) + s_pos*max(A_t,0)                            # Gottman et al. 2003 partner equation, RLS-identified, warm-started from a pooled DailyDialog fit
```

Cross-links into the rest of the state are exactly three: the threat buffer
(affect and mood only), an exploration multiplier by arousal band (request .6,
learn 1.0, explore 1.15), and the `bond` section of the actuation context
(request recommendation, expressed-valence steering with a bounded accommodation
cap and repair trigger, coordination excess against a published human band,
reciprocity debt and next disclosure depth). The bond never changes
reliability, authority, epistemic uncertainty or permissions; `act('request')`
consumes the partner's attention budget rather than the agent's resources, a
pending request blocks new ones until answered or timed out, and requests stop
being recommended once measured responsiveness falls below `request_floor`
(avoidant adaptation, so an unresponsive partner is not pestered up to the
budget). `neutral_updates_arousal` chooses whether goal-irrelevant,
unthreatening occurrences pull arousal down (Eq. 3 with zero stimulation) or
leave it to elapsed-time decay. `express()`
journals the agent's own expression as an action, not as evidence. All new
operations replay exactly. See [bonding report](bonding.md) and
[disclosure and safeguards](bonding_disclosure.md).

## Persistence, time, replay and isolation

All mutations use `BEGIN IMMEDIATE`, reload the current per-agent state, validate, append a hash-linked journal entry and update materialized state in one transaction. Distinct agent IDs isolate events, memories, outcomes and adaptive parameters even with identical event IDs. Hashes detect accidental journal inconsistency, not authenticated malicious tampering. External raw model/tool records can be retained with `record_artifact`; auth headers are excluded. Observed failure responses remain in `ProviderError.record` and can be explicitly journaled after rollback.

- **Wall time:** audit timestamps only; a process sleeping/restarting does not silently decay state.
- **Simulation time:** only `advance_time(dt)` advances explicit decay/recovery; negative/nonfinite dt is rejected.
- **Event time:** ordered journal mutations drive discrete transitions. Event timestamps remain evidence metadata, not reorder instructions.
- **Snapshot:** full evidence, memories, goals, time, config, trained weights and adaptation; digest checked on restore.
- **Restore/reset:** explicit journal transitions. Restore preserves all snapshot values; reset starts fresh state and requires an explicit operation.
- **Fork:** new agent ID, deep value copy of full snapshot, parent unchanged. Targets must be unused.
- **Replay:** default re-executes numerical transitions with recorded appraisals, outcomes, actions, retrieval parameters, configuration, artifacts and interventions, checking every recorded poststate. No provider call occurs. `mode='playback'` is an explicit lighter alternative that restores validated poststates. Live model re-execution is a different experiment and cannot promise bitwise equivalence.

SQLite transactions keep per-operation state atomic. A provider call currently runs inside the observation transaction for coherent appraisal; long calls hold the writer lock. Logical full-poststate semantics now use lossless compact journal deltas with an optional Rust kernel. Sparse histories retain much less disk data, while dense elapsed-time updates can still have quadratic changed-leaf volume. Snapshots use latest-hash lookup and replay streams entries; explicit full `trace()`/inspector views still materialize history. See [memory profiles](rust_memory.md). These deliberate research-scale limits require incremental tables and optimistic concurrency for large multi-agent deployments. Snapshot formats are experimental v1, with no cross-version migration yet.
