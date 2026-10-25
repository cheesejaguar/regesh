"""Durable cognitive loop. Every mutation is an atomic, inspectable transition."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from .schema import Event, canonical, finite
from .storage import Store, digest
from .appraisal import ReferenceAppraisal, validate_appraisal, clip
from .dynamics import initial_state, advance, transition, controls


class Agent:
    def __init__(self, db_path, agent_id="x", config=None, appraisal_provider=None, _create_only=False):
        if not isinstance(agent_id,str) or not agent_id: raise ValueError("agent_id required")
        self.store=Store(db_path)
        self.id=agent_id
        self.provider=appraisal_provider or ReferenceAppraisal()
        self.store.db.execute("BEGIN IMMEDIATE")
        try:
            existing=self.store.db.execute("SELECT 1 FROM agents WHERE id=?",(self.id,)).fetchone()
            if not existing:
                state=initial_state(config or {})
                self._validate_state(state)
                self.store.db.execute("INSERT INTO agents VALUES(?,?)",(self.id,canonical(state)))
                self._append("init",{"config":state["config"]},state)
            elif _create_only:
                raise ValueError("agent already exists")
            elif config and any(self.get_state()["config"].get(k)!=v for k,v in config.items()):
                raise ValueError("persisted configuration differs; fork or reset explicitly")
            self.store.db.execute("COMMIT")
        except BaseException:
            self.store.db.execute("ROLLBACK")
            raise

    def record_artifact(self,kind,payload):
        """Journal exact model/tool artifacts without treating them as evidence."""
        canonical(payload)
        def fn(s):
            return {"kind":kind,"artifact":payload},payload,True
        return self._mutate("artifact",fn)

    def close(self):
        self.store.close()

    def get_state(self):
        return self.store.read(self.id)

    def _append(self, operation, payload, state):
        row=self.store.db.execute("SELECT seq,hash FROM journal WHERE agent=? ORDER BY seq DESC LIMIT 1",(self.id,)).fetchone()
        entry={"seq":row[0]+1 if row else 0,"operation":operation,"payload":payload,"state":state,"previous":row[1] if row else ""}
        self.store.db.execute("INSERT INTO journal VALUES(?,?,?,?,?,?,?,?)",(self.id,entry["seq"],operation,canonical(payload),canonical(state),entry["previous"],digest(entry),datetime.now(timezone.utc).isoformat()))
        self.store.db.execute("UPDATE agents SET state=? WHERE id=?",(canonical(state),self.id))

    def _mutate(self, operation, fn):
        self.store.db.execute("BEGIN IMMEDIATE")
        try:
            state=self.get_state()
            payload,result,changed=fn(state)
            if changed:
                self._validate_state(state)
                self._append(operation,payload,state)
            self.store.db.execute("COMMIT")
            return deepcopy(result)
        except BaseException:
            self.store.db.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_state(state):
        canonical(state)
        if state["schema_version"]!=1:
            raise ValueError("unsupported state schema")
        finite(state["time"],0)
        for key in ("fast_tau","mood_tau","relationship_tau","disposition_tau"):
            finite(state["config"][key],.001)
        for k,v in state["internal"].items():
            finite(v,0,64 if k=="commitments" else 1)
        for v in state["goals"].values():
            finite(v,-1,1)
        for k,v in state["mood"].items():
            finite(v,-1 if k=="valence" else 0,1)
        for a in state["affect"].values():
            for k in ("satisfaction","threat","curiosity"):
                finite(a[k],0,1)
        for v in state["dispositions"].values():
            finite(v,-1,1)

    def appraise(self,event):
        return validate_appraisal(self.provider.appraise(event,self.get_state()))

    def observe(self,event):
        if isinstance(event,dict):
            event=Event(**event)
        # Re-validate nested mutable dataclass values at API boundary.
        event=Event(**event.to_dict())
        def fn(s):
            raw=event.to_dict()
            if event.id in s["events"]:
                if canonical(raw)!=canonical(s["events"][event.id]):
                    raise ValueError("event identity reused with different evidence")
                return {},s["memory"][event.id]["interpretations"][-1]["appraisal"],False
            appraisal=validate_appraisal(self.provider.appraise(event,deepcopy(s)))
            if any(k not in s["events"] and k!=event.id for k in appraisal["support"]):
                raise ValueError("appraisal cites unrecorded evidence")
            s["events"][event.id]=raw
            transition(s,event,appraisal)
            affect=s["affect"].get(event.id,{})
            s["memory"][event.id]={"observation":raw,"outcome":None,"interpretations":[{"version":1,"appraisal":appraisal,"reason":"initial"}],
                "encoded_salience":abs(appraisal["gain"])+appraisal["surprise"],"affect":deepcopy(affect),"retrievals":0}
            return {"event":raw,"appraisal":appraisal},appraisal,True
        return self._mutate("observe",fn)

    def advance_time(self,dt):
        finite(dt,0)
        def fn(s):
            advance(s,dt)
            return {"dt":dt},s["time"],True
        return self._mutate("advance_time",fn)

    def set_regulator(self, predictor):
        """Persist shared trained weights separately from agent adaptation."""
        model={"weights":predictor.weights.tolist(),"ridge":predictor.ridge,"metadata":predictor.metadata}
        def fn(s):
            s["config"]["regulator"]={**model,"digest":digest(model)}
            return {"model":s["config"]["regulator"]},None,True
        return self._mutate("set_regulator",fn)

    def set_task_context(self, context):
        from .learning import CONTEXT_NAMES
        if set(context)-set(CONTEXT_NAMES): raise ValueError("unknown task measurement")
        for value in context.values(): finite(value)
        def fn(s):
            s["task_context"]=deepcopy(context)
            return {"context":context},None,True
        return self._mutate("set_task_context",fn)

    def _controls(self, state):
        reference=controls(state)
        model=state["config"].get("regulator")
        if model is None: return reference
        from .learning import CounterfactualRegulator, RidgePredictor
        i=state["internal"]
        context={"uncertainty":i["uncertainty"],"load":i["burden"],
          "mood":state["mood"]["valence"],"budget_fraction":1-i["resources"],
          "resource_limit":i["resources"],**state.get("task_context",{})}
        regulator=CounterfactualRegulator(RidgePredictor(model["weights"],model["ridge"],model["metadata"]))
        learned=regulator.actuation_context(context)
        if not state["config"]["memory_coupling"]: learned["retrieval_bias"]=0.
        return {**reference,**learned,"model_digest":model["digest"],"measured_context":context}

    def actuation_context(self):
        return self._controls(self.get_state())

    def act(self,action,reason=None):
        """Execute a cognitive allocation, with real resource costs in this profile."""
        costs={"diverge":.12,"refine":.08,"combine":.1,"explore":.12,"verify":.08,"plan":.1,"retrieve":.03,"recover":0.,"resolve":.04,"commit":.02}
        if action=="auto":
            action=self.actuation_context().get("strategy","explore")
        if action not in costs:
            raise ValueError("unsupported cognitive action")
        def fn(s):
            i=s["internal"]
            if costs[action]>i["resources"]+1e-12:
                raise ValueError("resource budget exhausted; recover or advance time")
            i["resources"]=max(0.,i["resources"]-costs[action])
            i["burden"]=min(1.,i["burden"]+costs[action])
            if action=="recover":
                i["resources"]=min(1.,i["resources"]+.25)
                i["burden"]*=.5
            elif action=="verify":
                i["burden"]*=.9  # allocating review is not itself new factual evidence
            elif action=="resolve":
                i["commitments"]=max(0.,i["commitments"]-1.)
            elif action=="commit":
                if i["commitments"]>=64:
                    raise ValueError("commitment capacity exhausted")
                i["commitments"]+=1.
            context=self._controls(s)
            item={"action":action,"cost":costs[action],"time":s["time"],"reason":reason or context["reason"],"control":context}
            s["actions"].append(item)
            return item,item,True
        return self._mutate("act",fn)

    def record_outcome(self,event_id,quality,*,outcome_id=None,approval=0.,partner_correct=None,control=None,provenance="environment"):
        finite(quality,-1,1); finite(approval,-1,1)
        if control is not None: finite(control,0,1)
        if partner_correct is not None and not isinstance(partner_correct,bool):
            raise ValueError("partner_correct must be independently checked boolean")
        oid=outcome_id or event_id+":outcome"
        outcome={"id":oid,"event_id":event_id,"quality":quality,"approval":approval,"partner_correct":partner_correct,"control":control,"provenance":provenance}
        def fn(s):
            if oid in s["outcomes"]:
                if canonical(outcome)!=canonical({k:s["outcomes"][oid][k] for k in outcome}):
                    raise ValueError("outcome identity reused")
                return {},s["outcomes"][oid],False
            e=s["events"][event_id]
            if e["kind"]!="occurrence":
                raise ValueError("recall, predictions and quotations cannot produce observed outcomes")
            if s["memory"][event_id]["outcome"] is not None:
                raise ValueError("event already has outcome; add correcting evidence and reappraise")
            prediction=s["memory"][event_id]["interpretations"][-1]["appraisal"]["expected_quality"]
            outcome["prediction_error"]=quality-prediction
            s["outcomes"][oid]=outcome
            s["memory"][event_id]["outcome"]=oid
            if not s["config"]["frozen"]:
                ex=s["expectations"].setdefault(e["task"],{"mean":0.,"control":.5,"n":0})
                rate=self._controls(s)["learning_rate"]
                ex["mean"]+=rate*(quality-ex["mean"])
                ex["n"]+=1
                if control is not None: ex["control"]+=rate*(control-ex["control"])
                i=s["internal"]
                i["reliability"]=.9*i["reliability"]+.1*(quality+1)/2
                i["commitments"]=max(0.,i["commitments"]-1.)
                i["uncertainty"]=.9*i["uncertainty"]+.1*min(1.,abs(quality-prediction)/2)
                s["dispositions"]["precision"]=.98*s["dispositions"]["precision"]+.02*min(1.,abs(quality-prediction)/2)
                s["dispositions"]["optimism"]=.99*s["dispositions"]["optimism"]+.01*quality
                if partner_correct is not None:
                    r=s["relationships"][e["actor"]]
                    r["correct" if partner_correct else "incorrect"]+=1
                    r["reliability"]=r["correct"]/(r["correct"]+r["incorrect"])
            return {"outcome":outcome},outcome,True
        return self._mutate("record_outcome",fn)

    def reappraise(self,event_id,*,evidence_ids,reason="new evidence"):
        """Version interpretation; preserve original evidence and all learned outcomes."""
        def fn(s):
            original=Event(**s["events"][event_id])
            if not evidence_ids or any(k not in s["events"] or s["events"][k]["kind"]!="occurrence" for k in evidence_ids):
                raise ValueError("reappraisal requires recorded external evidence")
            d=deepcopy(original.data)
            # Latest explicit goal evidence may revise appraisal, not historical outcome.
            for k in evidence_ids:
                d.update({key:v for key,v in s["events"][k]["data"].items() if key in {"goal_delta","quality","control","uncertainty"}})
            revised=Event(**{**original.to_dict(),"data":d})
            a=validate_appraisal(self.provider.appraise(revised,deepcopy(s)))
            a["support"]=list(dict.fromkeys([event_id]+evidence_ids))
            versions=s["memory"][event_id]["interpretations"]
            versions.append({"version":len(versions)+1,"appraisal":a,"reason":reason})
            # Replace the event-specific state; do not count an occurrence again.
            recalled=Event(**{**revised.to_dict(),"kind":"reflection"})
            transition(s,recalled,a)
            return {"event_id":event_id,"evidence_ids":evidence_ids,"appraisal":a,"reason":reason},a,True
        return self._mutate("reappraise",fn)

    def retrieve(self,task=None,*,limit=5,evidence_only=False,reactivate=False):
        if not isinstance(limit,int) or limit<0: raise ValueError("invalid limit")
        def fn(s):
            bias=0. if evidence_only else self._controls(s)["retrieval_bias"]
            scored=[]
            for key,m in s["memory"].items():
                if evidence_only and m["observation"]["kind"]!="occurrence": continue
                match=1. if task and m["observation"]["task"]==task else 0.
                a=m["interpretations"][-1]["appraisal"]
                # All observations remain available through the neutral evidence path.
                score=match+(0. if evidence_only else .2*m["encoded_salience"]+.5*bias*a["gain"])
                scored.append((score,key))
            selected=[k for _,k in sorted(scored,key=lambda x:(-x[0],x[1]))[:limit]]
            result=[deepcopy(s["memory"][k]) for k in selected]
            if reactivate:
                for key in selected:
                    m=s["memory"][key]; m["retrievals"]+=1
                    e=Event(**{**m["observation"],"kind":"recall"})
                    transition(s,e,m["interpretations"][-1]["appraisal"])
            return {"selected":selected,"task":task,"limit":limit,"evidence_only":evidence_only,"reactivate":reactivate,"bias":bias},result,True
        return self._mutate("retrieve",fn)

    def intervene(self,changes,reason):
        """Explicit researcher intervention, never disguised as observed evidence."""
        allowed={"mood","internal","goals","dispositions"}
        if not reason or not isinstance(changes,dict) or set(changes)-allowed:
            raise ValueError("interventions need reason and known state fields")
        def fn(s):
            for key,value in changes.items():
                if not isinstance(value,dict): raise ValueError("intervention values must be mappings")
                if key!="goals" and set(value)-set(s[key]): raise ValueError("unknown state field")
                s[key].update(value)
            return {"changes":changes,"reason":reason},s,True
        return self._mutate("intervene",fn)

    def snapshot(self):
        s=self.get_state()
        return {"schema_version":1,"state":s,"digest":digest(s),"provenance":{"agent":self.id,"journal_hash":self.store.entries(self.id)[-1]["hash"]}}

    def restore(self,snapshot):
        if snapshot.get("schema_version")!=1 or digest(snapshot["state"])!=snapshot["digest"]:
            raise ValueError("snapshot integrity/schema failure")
        restored=deepcopy(snapshot["state"])
        self._validate_state(restored)
        def fn(s):
            s.clear(); s.update(restored)
            return {"snapshot":snapshot},s,True
        return self._mutate("restore",fn)

    def fork(self,new_agent_id,snapshot=None):
        if self.store.db.execute("SELECT 1 FROM agents WHERE id=?",(new_agent_id,)).fetchone():
            raise ValueError("fork target exists")
        path=self.store.db.execute("PRAGMA database_list").fetchone()[2]
        if not path: raise ValueError("fork requires a file-backed database")
        child=Agent(path,new_agent_id,appraisal_provider=self.provider,_create_only=True)
        child.restore(snapshot or self.snapshot())
        return child

    def reset(self,config=None):
        def fn(s):
            cfg=config if config is not None else s["config"]
            s.clear(); s.update(initial_state(cfg))
            return {"config":cfg},s,True
        return self._mutate("reset",fn)

    def trace(self):
        return self.store.entries(self.id)

    def replay(self,new_agent_id,through=None,mode="recompute"):
        """Replay exact recorded appraisals, timing, actions, interventions and outcomes.

        Default recomputes transitions and checks every recorded poststate. The
        explicit `playback` mode only restores hash-validated recorded poststates.
        Neither mode calls a live provider. Live re-execution is a new experiment.
        """
        entries=self.trace()
        chosen=entries if through is None else [e for e in entries if e["seq"]<=through]
        if not chosen: raise ValueError("empty replay")
        if mode=="playback":
            return self.fork(new_agent_id,{"schema_version":1,"state":chosen[-1]["state"],"digest":digest(chosen[-1]["state"]),"provenance":{"replay_source":self.id}})
        if mode!="recompute": raise ValueError("unknown replay mode")
        if self.store.db.execute("SELECT 1 FROM agents WHERE id=?",(new_agent_id,)).fetchone():
            raise ValueError("replay target exists")
        path=self.store.db.execute("PRAGMA database_list").fetchone()[2]
        if not path: raise ValueError("replay requires a file-backed database")
        child=Agent(path,new_agent_id,chosen[0]["state"]["config"],_create_only=True)
        class RecordedAppraisal:
            def __init__(self,value): self.value=value
            def appraise(self,*args): return deepcopy(self.value)
        for entry in chosen:
            op,p=entry["operation"],entry["payload"]
            if op=="init": pass
            elif op=="observe":
                child.provider=RecordedAppraisal(p["appraisal"]);child.observe(Event(**p["event"]))
            elif op=="advance_time": child.advance_time(p["dt"])
            elif op=="act": child.act(p["action"],reason=p["reason"])
            elif op=="record_outcome":
                o=p["outcome"]
                child.record_outcome(o["event_id"],o["quality"],outcome_id=o["id"],
                    **{k:o[k] for k in ("approval","partner_correct","control","provenance")})
            elif op=="reappraise":
                child.provider=RecordedAppraisal(p["appraisal"])
                child.reappraise(p["event_id"],evidence_ids=p["evidence_ids"],reason=p["reason"])
            elif op=="retrieve":
                child.retrieve(p["task"],limit=p["limit"],evidence_only=p["evidence_only"],reactivate=p["reactivate"])
            elif op=="intervene": child.intervene(p["changes"],p["reason"])
            elif op=="restore": child.restore(p["snapshot"])
            elif op=="reset": child.reset(p["config"])
            elif op=="artifact": child.record_artifact(p["kind"],p["artifact"])
            elif op=="set_task_context": child.set_task_context(p["context"])
            elif op=="set_regulator":
                from .learning import RidgePredictor
                m=p["model"];child.set_regulator(RidgePredictor(m["weights"],m["ridge"],m["metadata"]))
            else: raise ValueError("unsupported replay operation: "+op)
            if digest(child.get_state())!=digest(entry["state"]):
                child.close()
                raise ValueError(f"recomputed state mismatch at sequence {entry['seq']}")
        child.provider=self.provider
        return child
