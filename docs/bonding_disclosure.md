# Pair-bonding mechanisms: novelty statement, candidate claims and safeguards

Date: 2026-10-25. This document separates what is borrowed from the literature,
what is an engineering synthesis, and what we believe is a specific new
mechanism. It is **not** a legal opinion. Whether any item is patentable requires
a professional prior-art search and legal analysis; software, abstract-idea and
obviousness limits vary by jurisdiction, and a public repository is itself a
disclosure that starts statutory clocks in some jurisdictions. Nothing here
establishes subjective experience, biological fidelity or benefit to users.

## Established precedent (not claimed)

| Mechanism | Primary sources | Used here as |
|---|---|---|
| Arousal regulation with a caregiver's comfort; "needy"/"independent" profiles; adaptation of the profile to caregiver responsiveness | Hiolle, Cañamero, Davila-Ross & Bard (2012, ACM TiiS 2(1):3); Hiolle, Lewis & Cañamero (2014, Front. Neurorobot. 8:17) | Equations (3)-(7) and Tables 1-2, implemented verbatim in `regesh/bonding/attachment.py` |
| Attachment behavioural system: proximity maintenance, safe haven, secure base; responsiveness as the antecedent of security | Bowlby; Ainsworth et al. (1978); De Wolff & van IJzendoorn (1997) | Design rationale for help-seeking and security |
| Social buffering of threat by a close partner, scaled by relationship quality | Coan, Schaefer & Davidson (2006, Psychol. Sci.); Coan & Sbarra (2015) | Multiplicative threat buffer gated by measured security and presence |
| Coupled difference equations for dyadic affect with inertia, uninfluenced steady states, influence functions, null clines and attractors; repair and damping terms | Cook et al. (1995); Gottman, Swanson & Murray (1999); Gottman et al. (2002, 2003) | `regesh/bonding/dyad.py` model, estimation algorithm and published parameter table |
| Mid-range (not maximal) interactive contingency predicts secure attachment | Jaffe, Beebe, Feldstein, Crown & Jasnow (2001); Beebe et al. (2010) | Rationale for capping accommodation and coordination |
| Rupture and repair in dyadic interaction | Tronick & Cohn (1989); Gottman et al. (2003) | Repair trigger in steering |
| Linguistic style coordination measure and its asymmetry; language style matching predicts relationship initiation/stability | Danescu-Niculescu-Mizil, Gamon & Dumais (2011); Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg (2012); Ireland et al. (2011); Niederhoffer & Pennebaker (2002) | Measures in `regesh/bonding/synchrony.py` |
| LLMs over-converge to users' style relative to human-human baselines | Blevins (2026, arXiv:2605.29278); French, D'Mello & von der Wense (2024) | Human-range reference band for the coordination control |
| Intimacy as an interpersonal process: self-disclosure, partner disclosure and perceived responsiveness; emotional disclosure matters more than factual | Reis & Shaver (1988); Laurenceau, Feldman Barrett & Pietromonaco (1998) | Published standardized paths as relative weights of the intimacy update |
| Reciprocal, escalating, turn-taking self-disclosure increases closeness and liking, including from computers | Aron et al. (1997); Moon (2000); Sprecher et al. (2013) | Graduated reciprocity recommendation |
| Relational agents maintain long-term human-computer relationships; social chatbots become attachment objects | Bickmore & Picard (2005); Xie & Pentina (2022); Pentina, Hancock & Xie (2023); De Freitas (2026, arXiv:2606.20589) | Prior art for the application; source of the disengagement safeguard |
| Persistent affect platforms with relationships, needs and emotion | Humanoid Agents (Wang et al. 2023); Sentipolis (Fu et al. 2026); FAtiMA; WASABI; see `research_matrix.md` | Baseline architectures; none evaluated against the bonding literature above |

## Learned components (evaluated, mostly engineering)

Emotion classification from language-model embeddings and sequence models of
dialogue affect are abundant prior art and are not claimed. What was measured
here: a logistic head on the generating model's embeddings beats the
repository's hashed-token reference on GoEmotions and transfers its valence
sign to conversation turns; a learned partner model adds nothing beyond
scalar affect features; the online identifier's cold start, exposed by that
comparison, is fixed with a corpus-derived warm start. The only mechanism-level
elements are the neutral gate with recorded provenance and the rule that the
sensor is a host-side evidence provider whose outputs are journaled as
interpretations, never as facts.

## Engineering synthesis (ours, not asserted as novel in isolation)

A replayable, hash-journaled bond state per partner that combines the three
literatures above inside one cognitive-control loop; shared pure-function
modules used both by the platform and by the independent paper reproductions,
so the mechanism evaluated is the mechanism deployed.

## Candidate specific mechanisms

Each item names the closest prior art and the specific departure. Patentability
is not asserted.

1. **Evidence-contingent responsiveness with a trust firewall.** Responsiveness is
   updated only when a journaled support request is resolved by a later,
   independently recorded partner occurrence (or times out), rather than from
   sensor presence. The resulting security value may modulate threat, arousal and
   exploration but is structurally prevented from altering factual reliability,
   authority, epistemic uncertainty or tool permissions. Closest prior art: Hiolle
   et al. (2014) Eq. (5) updates from perceived comfort each timestep; Humanoid
   Agents couple closeness to conversation choices. Departure: idempotent
   request/response accounting, replayability, and the explicit firewall. The
   estimator may be the published per-resolution update or a Beta evidence-count
   posterior, which the platform simulation showed is needed for responsiveness to
   track partner behaviour within a few hundred turns.
2. **Attention-budgeted help-seeking.** A support request consumes a per-partner
   attention budget that recovers only with elapsed time; the recommendation is
   suppressed when the budget is exhausted, and a pending request blocks new ones.
   Closest prior art: robot "Find-Human" behaviour triggered purely by arousal
   (Hiolle); companion apps using simulated distress to retain users (De Freitas
   2026, as a documented harm). Departure: a hard budget that bounds the rate of
   affective appeals toward the user, a responsiveness floor below which requests
   stop (avoidant adaptation rather than escalation), and inspectability of both.
3. **Security-gated, presence-weighted threat buffer.** Raw goal-loss threat is
   multiplied by (1 - gain * responsiveness * familiarity * presence), so a
   stranger buffers little and an absent partner nothing; the buffer changes
   affect and mood only. Closest prior art: Coan et al. (2006) human finding;
   Regesh's own relationship affiliation variable. Departure: a computed buffer from
   measured interaction history, firewalled from epistemics.
4. **Online identification of the partner's affect equation with bounded
   attractor steering.** A recursive least-squares identifier estimates the
   partner's inertia, uninfluenced constant and bilinear influence slopes from
   the (partner affect, agent expressed affect, next partner affect) sequence;
   a one-step model-predictive choice of expressed valence steers toward a
   positive target but is capped around the agent's own valence so the agent
   cannot mirror the partner fully; a repair jump triggers at a negativity
   threshold; the closed-loop fixed point is reported. Closest prior art:
   Gottman's offline estimation and clinical "repair" concept; mirroring/empathy
   in relational agents (Bickmore & Picard 2005). Departure: online estimation,
   explicit accommodation cap motivated by mid-range contingency, and exposure of
   the predicted attractor as a control variable. The identifier is warm-started
   from a corpus-derived prior; a learned embedding-based alternative was
   evaluated and did not improve on it (`bonding_partner_model.md`).
5. **Mid-range coordination control.** Function-word coordination toward the
   partner is measured online (Eq. 1 of Danescu-Niculescu-Mizil et al. 2012) and
   compared with a published human-human band; the actuation context reports the
   excess, so a host can damp over-accommodation. Closest prior art: alignment
   measurement papers (French et al. 2024; Blevins 2026) measure but do not
   control. Departure: closed-loop use of the measure with a target band.
6. **Reciprocity-weighted intimacy with graduated disclosure.** Intimacy is
   updated with published standardized path weights (self .22, partner .32,
   responsiveness .36; emotional disclosure only) and the recommendation is to
   disclose next at the partner's depth plus a bounded step only when a
   reciprocity debt exists. Closest prior art: Moon (2000) fixed disclosure
   scripts; Aron et al. (1997) fixed escalation. Departure: evidence-driven,
   bounded, turn-taking reciprocity inside an affect loop.

## Safeguards (part of the mechanism, not an afterthought)

- Bond state never grants or raises authority, reliability or permissions
  (tested in `tests/test_bonding.py`).
- Requests are budgeted by the partner's attention, time out, and never fire on
  disengagement events; the platform emits no guilt appeals or simulated distress
  at disengagement. These are the "caregiving-system capture" tactics documented
  by De Freitas (2026), and they are excluded by construction.
- Coordination is targeted to a human-range band, not maximised, because
  heightened contingency predicts insecure attachment (Jaffe et al. 2001) and
  LLM over-convergence is already excessive (Blevins 2026).
- All bond variables are inspectable, interveneable (`intervene({'bonds': ...})`),
  resettable and replayable; interventions are journaled as interventions, not
  disguised as evidence.
- Measured responsiveness and intimacy are functional variables about recorded
  interaction, not inferences about the user's feelings, traits or clinical state.

## What would falsify usefulness

A matched comparison with real users is required before any claim that these
mechanisms improve collaboration or wellbeing. The present evidence is limited
to simulation checks, reproductions of published computational models, a
measurement-validation reproduction on public conversation data, and a small
scoped real-model replication; see `bonding.md`.
