"""Transparent reference dynamics, separated from storage and model providers."""
import math
from .appraisal import clip


def initial_state(config):
    defaults={"seed":0,"model":"reference-v1","memory_coupling":True,"frozen":False,
              "fast_tau":3.,"mood_tau":24.,"relationship_tau":240.,"disposition_tau":1000.}
    defaults.update(config)
    return {"schema_version":1,"config":defaults,"time":0.,"goals":{"quality":1.},
      "affect":{},"mood":{"valence":0.,"arousal":0.},"relationships":{},
      "expectations":{},"dispositions":{"precision":.5,"optimism":0.},
      "internal":{"resources":1.,"commitments":0.,"reliability":.9,"uncertainty":.5,"burden":0.},
      "events":{},"memory":{},"outcomes":{},"actions":[],"habituation":{},"last_appraisal":None}


def advance(state, dt):
    state["time"]+=dt
    cfg=state["config"]
    for a in state["affect"].values():
        for key in ("satisfaction","threat","curiosity"):
            a[key]*=math.exp(-dt/cfg["fast_tau"])
    for k in state["mood"]:
        state["mood"][k]*=math.exp(-dt/cfg["mood_tau"])
    # Affiliation decays towards neutral; factual reliability is not erased by time.
    for r in state["relationships"].values():
        r["affiliation"]*=math.exp(-dt/cfg["relationship_tau"])
    state["dispositions"]["optimism"]*=math.exp(-dt/cfg["disposition_tau"])
    for k in state["habituation"]:
        state["habituation"][k]*=math.exp(-dt/12.)
    internal=state["internal"]
    internal["burden"]*=math.exp(-dt/8.)
    internal["resources"]=min(1.,internal["resources"]+dt*.025)


def transition(state,event,appraisal):
    if state["config"]["frozen"]:
        return
    a=appraisal
    occurring=event.kind=="occurrence"
    intensity=event.confidence*(1. if occurring else .15)
    signature=event.task+":"+event.target+":"+str(round(a["gain"],1))
    repetitions=state["habituation"].get(signature,0.)
    intensity/=1+.3*repetitions
    if occurring:
        state["habituation"][signature]=repetitions+1
    gain=a["gain"]
    # High burden alone never produces threat. Goal loss and low control do.
    threat=max(0.,-gain)*(1-.65*a["control"])*(1+.2*state["internal"]["uncertainty"])
    affect={"satisfaction":max(0.,gain)*intensity,"threat":min(1.,threat*intensity),
            "curiosity":a["surprise"]*a["control"]*intensity,"target":event.target,"time":state["time"]}
    state["affect"][event.id]=affect
    state["mood"]["valence"]=clip(.88*state["mood"]["valence"]+.12*(affect["satisfaction"]-affect["threat"]))
    state["mood"]["arousal"]=clip(.85*state["mood"]["arousal"]+.15*(affect["threat"]+affect["curiosity"]),0,1)
    state["last_appraisal"]=a
    if occurring:
        r=state["relationships"].setdefault(event.actor,{"familiarity":0.,"affiliation":0.,"reliability":.5,"authority":0.,"correct":1.,"incorrect":1.})
        r["familiarity"]=min(1.,r["familiarity"]+.05)
        r["affiliation"]=clip(.95*r["affiliation"]+.05*a["approval"])
        state["internal"]["uncertainty"]=.8*state["internal"]["uncertainty"]+.2*a["uncertainty"]


def controls(state):
    a=state["last_appraisal"] or {"control":.5,"surprise":0.,"uncertainty":.5}
    valence=state["mood"]["valence"]
    arousal=state["mood"]["arousal"]
    i=state["internal"]
    exploration=clip(.35+.3*valence+.35*a["control"]+.2*a["surprise"]-.3*i["uncertainty"],0,1)
    verification=clip(.2+.4*i["uncertainty"]+.3*arousal+.1*state["dispositions"]["precision"]+.15*(1-i["reliability"]),0,1)
    return {"exploration":exploration,"verification":verification,
      "retrieval_bias":valence if state["config"]["memory_coupling"] else 0.,
      "learning_rate":.08+.24*a["surprise"],
      "planning_horizon":max(1,round((2+6*a["control"]*i["resources"])/(1+.03*i["commitments"]+.5*i["burden"]))),
      "available_resources":i["resources"],"evidence_standard":"observed outcomes, independent of approval",
      "reason":{"mood":dict(state["mood"]),"uncertainty":i["uncertainty"],"control":a["control"]}}
