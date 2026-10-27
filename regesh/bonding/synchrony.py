"""Linguistic style coordination and matching as a measured bonding channel.

Measures follow published definitions:

* Coordination of speaker b towards target a on marker m (Danescu-Niculescu-Mizil,
  Gamon & Dumais, WWW 2011; Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg,
  WWW 2012, Eq. 1): C^m(b,a) = P(E^m_{u2} | E^m_{u1}) - P(E^m_{u2}), estimated
  over exchanges (a: u1, b: u2), defined only when a exhibits m at least once.
* Language style matching (Ireland et al., Psychological Science 2011; Gonzales,
  Hancock & Pennebaker 2010): per category 1 - |p_a - p_b| / (p_a + p_b + .0001),
  averaged over categories.

The function-word lexicon below is authored here from closed-class English
vocabulary. It is **not** the proprietary LIWC dictionary used by the original
papers, so absolute levels are not comparable; directions and contrasts are.
"""
from __future__ import annotations
import math
import re

CATEGORIES = ("ppron", "ipron", "article", "conj", "prep", "auxverb", "adverb", "negate", "quant")
# The eight markers used by Danescu-Niculescu-Mizil et al. (2012) exclude negations.
COORDINATION_MARKERS = tuple(c for c in CATEGORIES if c != "negate")

LEXICON = {
    "ppron": """i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his
        himself she her hers herself they them their theirs themselves i'm i've i'd i'll you're you've
        you'd you'll we're we've we'd we'll he's he'd he'll she's she'd she'll they're they've they'd
        they'll ya y'all thee thou thy thine""",
    "ipron": """it its itself it's that this these those what which who whom whose whatever whichever
        whoever something anything nothing everything somebody anybody nobody everybody someone anyone
        everyone one ones other others another that's there's here's what's who's""",
    "article": "a an the",
    "conj": """and but or nor so yet because cause cuz cos although though altho tho while whereas unless
        until till whether if then than as since once when whenever also plus either neither otherwise
        except""",
    "prep": """about above across after against along among amongst around at before behind below beneath
        beside besides between beyond by despite down during except for from in inside into like near
        of off on onto out outside over past per since through throughout till to toward towards under
        underneath until unto up upon via with within without atop amid amidst aboard versus vs
        regarding concerning minus""",
    "auxverb": """am is are was were be been being have has had having do does did doing can could will
        would shall should may might must ought isn't aren't wasn't weren't haven't hasn't hadn't don't
        doesn't didn't can't cannot couldn't won't wouldn't shan't shouldn't mightn't mustn't ain't
        gonna gotta i'm i've i'd i'll you're you've you'd you'll we're we've we'd we'll he's he'd he'll
        she's she'd she'll they're they've they'd they'll it's that's there's here's what's who's""",
    "adverb": """very really too quite rather pretty just only even still already almost nearly always
        never often sometimes usually seldom rarely ever again now then here there soon later maybe
        perhaps probably actually basically definitely certainly truly totally completely absolutely
        especially particularly mostly mainly generally simply exactly hardly barely somewhat kinda
        sorta literally seriously honestly apparently obviously clearly surely anyway anyways however
        therefore instead otherwise meanwhile afterwards how why""",
    "negate": """no not never none nothing nobody nowhere neither nor cannot isn't aren't wasn't weren't
        don't doesn't didn't won't wouldn't can't couldn't shouldn't haven't hasn't hadn't mustn't ain't
        nope nah without""",
    "quant": """all any both each every few fewer fewest less least little lot lots many more most much
        none several some enough plenty numerous various whole half double twice entire total bunch
        couple majority minority part tons ton loads dozens hundreds thousands millions zero single
        multiple extra additional remaining rest""",
}
LEXICON = {k: frozenset(v.split()) for k, v in LEXICON.items()}
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _TOKEN.findall(text.lower().replace("’", "'"))


def exhibits(text, category, tokens=None):
    """Whether an utterance contains at least one word of the marker category."""
    words = LEXICON[category]
    return any(t in words for t in (tokens if tokens is not None else tokenize(text)))


def marker_vector(text):
    tokens = tokenize(text)
    return {c: exhibits(text, c, tokens) for c in CATEGORIES}


def proportions(text):
    """Fraction of tokens in each category; empty text yields zeros."""
    tokens = tokenize(text)
    n = len(tokens)
    return {c: (sum(t in LEXICON[c] for t in tokens) / n if n else 0.) for c in CATEGORIES}


def lsm(text_a, text_b, categories=CATEGORIES):
    """Ireland et al. (2011) language style matching between two texts, in [0,1]."""
    pa, pb = proportions(text_a), proportions(text_b)
    return lsm_from_proportions(pa, pb, categories)


def lsm_from_proportions(pa, pb, categories=CATEGORIES):
    scores = [1 - abs(pa[c] - pb[c]) / (pa[c] + pb[c] + .0001) for c in categories]
    return sum(scores) / len(scores)


class CoordinationCounts:
    """Sufficient statistics for C^m(b,a) over a stream of (target u1, speaker u2) exchanges.

    Only booleans are stored, so the accumulator is JSON-serialisable and replayable.
    """
    def __init__(self, markers=COORDINATION_MARKERS):
        self.markers = tuple(markers)
        self.n = 0
        self.a = {m: 0 for m in self.markers}      # target exhibits m
        self.b = {m: 0 for m in self.markers}      # speaker reply exhibits m
        self.ab = {m: 0 for m in self.markers}     # both

    def add(self, target_text, reply_text):
        ta, tb = tokenize(target_text), tokenize(reply_text)
        self.n += 1
        for m in self.markers:
            ea, eb = exhibits("", m, ta), exhibits("", m, tb)
            self.a[m] += ea; self.b[m] += eb; self.ab[m] += ea and eb
        return self

    def coordination(self, m):
        """Eq. (1); None when the target never exhibited m (undefined)."""
        if self.a[m] == 0 or self.n == 0:
            return None
        return self.ab[m] / self.a[m] - self.b[m] / self.n

    def summary(self):
        values = {m: self.coordination(m) for m in self.markers}
        defined = [v for v in values.values() if v is not None]
        return {"exchanges": self.n, "by_marker": values,
                "aggregated_3": (sum(defined) / len(defined)) if defined else None,
                "aggregated_1": (sum(defined) / len(defined)) if len(defined) == len(self.markers) else None}

    def to_dict(self):
        return {"markers": list(self.markers), "n": self.n, "a": dict(self.a), "b": dict(self.b), "ab": dict(self.ab)}

    @classmethod
    def from_dict(cls, value):
        obj = cls(value["markers"]); obj.n = value["n"]
        obj.a, obj.b, obj.ab = dict(value["a"]), dict(value["b"]), dict(value["ab"])
        return obj


def coordination(exchanges, markers=COORDINATION_MARKERS):
    """Per-marker coordination of the replying speaker towards the target over exchanges."""
    counts = CoordinationCounts(markers)
    for target_text, reply_text in exchanges:
        counts.add(target_text, reply_text)
    return counts.summary()


def group_coordination(speaker_summaries, markers=COORDINATION_MARKERS):
    """Macro-averages across speakers; the three aggregations of Danescu-Niculescu-Mizil et al. (2012).

    speaker_summaries: iterable of CoordinationCounts.summary() values, one per speaker.
    """
    rows = list(speaker_summaries)
    by_marker = {}
    for m in markers:
        vals = [r["by_marker"][m] for r in rows if r["by_marker"].get(m) is not None]
        by_marker[m] = (sum(vals) / len(vals)) if vals else None
    complete = [r for r in rows if all(r["by_marker"].get(m) is not None for m in markers)]
    agg1 = (sum(sum(r["by_marker"][m] for m in markers) / len(markers) for r in complete) / len(complete)) if complete else None
    agg2_vals, agg3_vals = [], []
    for r in rows:
        if not any(r["by_marker"].get(m) is not None for m in markers):
            continue
        smoothed = [r["by_marker"][m] if r["by_marker"].get(m) is not None else by_marker[m] for m in markers]
        if all(v is not None for v in smoothed):
            agg2_vals.append(sum(smoothed) / len(markers))
        own = [r["by_marker"][m] for m in markers if r["by_marker"].get(m) is not None]
        agg3_vals.append(sum(own) / len(own))
    return {"speakers": len(rows), "speakers_complete": len(complete), "by_marker": by_marker,
            "aggregated_1": agg1,
            "aggregated_2": (sum(agg2_vals) / len(agg2_vals)) if agg2_vals else None,
            "aggregated_3": (sum(agg3_vals) / len(agg3_vals)) if agg3_vals else None}


# Human-human reference band for mean function-word convergence, from Blevins (2026,
# arXiv:2605.29278, Table 2, LIWC mean column): DailyDialog .036, Ubuntu .022/.020,
# WildChat users .035; the LLM in the same corpus reached .068. Mid-range coordination is the target, following
# Jaffe et al. (2001) and Beebe et al. (2010): more contingency is not necessarily better.
HUMAN_CONVERGENCE_REFERENCE = {"center": .036, "band": .03, "llm_overconvergence": .068}


def accommodation_assessment(mean_convergence, reference=HUMAN_CONVERGENCE_REFERENCE):
    """Where measured convergence sits relative to the published human-human band."""
    if mean_convergence is None or not math.isfinite(mean_convergence):
        return {"measured": None, "excess": None, "band": "undefined"}
    excess = mean_convergence - reference["center"]
    band = "within" if abs(excess) <= reference["band"] else ("over" if excess > 0 else "under")
    return {"measured": mean_convergence, "excess": excess, "band": band,
            "target_center": reference["center"], "target_halfwidth": reference["band"]}
