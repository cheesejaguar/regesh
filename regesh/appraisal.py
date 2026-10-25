"""Appraisal is an interpretation of evidence, never an authoritative intent label."""
from .schema import finite


def clip(x, lo=-1., hi=1.):
    return max(lo,min(hi,x))


class ReferenceAppraisal:
    identifier = "reference-appraisal-v1"

    def appraise(self, event, state):
        d=event.data
        weights=state["goals"]
        gain=sum(weights.get(k,0)*v for k,v in d.get("goal_delta",{}).items())
        relevance=min(1.,sum(abs(weights.get(k,0)*v) for k,v in d.get("goal_delta",{}).items()))
        expected=state["expectations"].get(event.task,{"mean":0.,"control":.5})
        quality=d.get("quality")
        surprise=abs(quality-expected["mean"])/2 if quality is not None else d.get("uncertainty",.5)*.5
        # Facts about praise and task quality stay separate. Neither infers intent.
        reliability=state["relationships"].get(event.actor,{"reliability":.5})["reliability"]
        return {
            "schema_version":1,"provider":self.identifier,
            "goal_relevance":relevance,"gain":clip(gain),"expected_quality":expected["mean"],
            "surprise":surprise,"control":d.get("control",expected["control"]),
            "responsibility":d.get("responsibility",0.),"approval":d.get("approval",0.),
            "uncertainty":d.get("uncertainty",1-reliability),
            "support":[event.id],"alternatives":d.get("alternatives",["intent unknown"]),
            "social_meaning":"reported approval" if "approval" in d else "unspecified",
        }


def validate_appraisal(value):
    for key in ("goal_relevance","surprise","control","responsibility","uncertainty"):
        finite(value[key],0,1)
    for key in ("gain","expected_quality","approval"):
        finite(value[key],-1,1)
    if not isinstance(value.get("support"),list) or not isinstance(value.get("alternatives"),list):
        raise ValueError("appraisal must retain evidence and alternatives")
    return value
