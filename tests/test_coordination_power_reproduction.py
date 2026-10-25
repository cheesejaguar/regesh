import gzip
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from regesh.bonding.synchrony import COORDINATION_MARKERS, CoordinationCounts
from regesh.reproductions import coordination_power as cp


def synthetic_counts(n=10, target_hits=4, extra_reply_hits=3):
    """Target exhibits 'article' in target_hits of n utterances; the reply echoes every one of
    those and additionally exhibits it in extra_reply_hits other replies, so
    C^article = 1 - (target_hits + extra_reply_hits) / n exactly. Other words are content-only."""
    counts = CoordinationCounts(COORDINATION_MARKERS)
    for i in range(n):
        target = "the court ruled" if i < target_hits else "court ruled"
        reply = "the ruling stands" if i < target_hits + extra_reply_hits else "ruling stands"
        counts.add(target, reply)
    return counts


def fixture_corpus(directory: Path) -> dict:
    """Two sessions of one case plus one session of another, ConvoKit file layout, inside a zip."""
    speakers = {"j__chief": {"name": "Chief", "type": "J", "role": "justice"},
                "j__assoc": {"name": "Assoc", "type": "J", "role": "justice"},
                "lawyer_a": {"name": "A", "type": "A", "role": "advocate"},
                "lawyer_b": {"name": "B", "type": "A", "role": "advocate"},
                "<INAUDIBLE>": {"name": "<INAUDIBLE>", "type": "U", "role": "inaudible"}}
    conversations = {
        "c1": {"case_id": "2004_01-100", "advocates": {"lawyer_a": {"side": 1, "role": "x"}, "lawyer_b": {"side": 0, "role": "y"}},
               "win_side": 1, "votes_side": {"j__chief": 1, "j__assoc": 0}},
        "c1b": {"case_id": "2004_01-100", "advocates": {"lawyer_a": {"side": 1, "role": "x"}, "lawyer_b": {"side": 0, "role": "y"}},
                "win_side": 1, "votes_side": {"j__chief": 1, "j__assoc": 0}},
        "c2": {"case_id": "2004_01-orig", "advocates": {"lawyer_a": {"side": 2, "role": "amicus"}},
               "win_side": 0, "votes_side": {"j__chief": 0, "j__assoc": 0}},
        "c3": {"case_id": "2004_09-999", "advocates": {"lawyer_b": {"side": 0, "role": "z"}},
               "win_side": 0, "votes_side": {"j__chief": 0, "j__assoc": 0}}}
    utterances = []

    def utt(conversation, index, speaker, text, reply_to):
        uid = f"{conversation}__{index:03d}"
        utterances.append({"id": uid, "conversation_id": conversation, "text": text, "speaker": speaker, "reply_to": reply_to,
                           "meta": {"case_id": conversations[conversation]["case_id"], "speaker_type": speakers[speaker]["type"]}})
        return uid

    rng = np.random.default_rng(3)
    words = ["the", "a", "we", "it", "and", "of", "is", "very", "some", "court", "case", "law", "argue", "rule", "statute"]
    for conversation in ("c1", "c1b", "c2", "c3"):
        previous = utt(conversation, 0, "j__chief", "We will hear argument in the case.", None)
        for index in range(1, 61):
            if index % 4 == 1:
                speaker = "lawyer_a" if conversation != "c3" else "lawyer_b"
            elif index % 4 == 2:
                speaker = "j__assoc" if index % 8 == 2 else "j__chief"
            elif index % 4 == 3:
                speaker = "lawyer_b" if conversation in ("c1", "c1b") else "lawyer_a"
            else:
                speaker = "j__chief" if index % 8 == 0 else "j__assoc"
            text = " ".join(rng.choice(words, size=int(rng.integers(3, 12))))
            if index == 30:
                text = ""  # one empty utterance is dropped with its reply
            previous = utt(conversation, index, speaker, text, previous)
        utt(conversation, 61, "<INAUDIBLE>", "inaudible", previous)
        utt(conversation, 62, "j__chief", "the end", f"{conversation}__061")
    term_dir = directory / "zipsrc" / "supreme-2004"
    term_dir.mkdir(parents=True)
    (term_dir / "speakers.json").write_text(json.dumps(speakers))
    (term_dir / "conversations.json").write_text(json.dumps(conversations))
    (term_dir / "utterances.jsonl").write_text("\n".join(json.dumps(u) for u in utterances) + "\n")
    cache = directory / "cache"
    cache.mkdir()
    with zipfile.ZipFile(cache / "supreme-2004.zip", "w") as archive:
        for name in ("utterances.jsonl", "speakers.json", "conversations.json"):
            archive.write(term_dir / name, f"supreme-2004/{name}")
    cases = [{"id": "2004_01-100", "votes_side": {"j__chief": 1.0, "j__assoc": 0.0},
              "advocates": {"A": {"id": "lawyer_a", "side": 1}, "B": {"id": "lawyer_b", "side": 0}}},
             {"id": "2004_01-orig", "votes_side": {"j__chief": 0.0, "j__assoc": 1.0}, "advocates": {"A": {"id": "lawyer_a", "side": 2}}}]
    (cache / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    inputs = {name: {"sha256": hashlib.sha256((cache / name).read_bytes()).hexdigest()} for name in ("supreme-2004.zip", "cases.jsonl")}
    config = {"schema_version": 1, "search_date": "test", "inputs": inputs, "terms": [2004],
              "case_ids": ["01-100", "01-100b", "01orig", "77-777"], "case_id_matching": "aliased",
              "case_id_aliases": {"01-100b": "01-100", "01orig": "01-orig"},
              "chief_justice_ids": ["j__chief"],
              "thresholds": {"primary": {"target_min_marker_utterances": 1, "speaker_min_marker_utterances": 0, "min_exchanges": 1},
                             "sensitivities": {"strict": {"target_min_marker_utterances": 5, "speaker_min_marker_utterances": 5, "min_exchanges": 1}}},
              "statistics": {"seed": 7, "bootstrap_samples": 200}, "published": {"claims": {}}}
    return {"cache": cache, "config": config, "utterances": utterances}


class CoordinationPowerReproductionTests(unittest.TestCase):
    def test_synthetic_conditional_probabilities_give_known_coordination(self):
        counts = synthetic_counts(n=10, target_hits=4, extra_reply_hits=3)
        self.assertAlmostEqual(counts.coordination("article"), 1 - 7 / 10)
        self.assertIsNone(counts.coordination("quant"))  # target never exhibits quantifiers
        loose = cp.thresholded_summary(counts, {"target_min_marker_utterances": 1, "speaker_min_marker_utterances": 0})
        self.assertAlmostEqual(loose["by_marker"]["article"], .3)
        self.assertIsNone(cp.thresholded_summary(counts, {"target_min_marker_utterances": 5, "speaker_min_marker_utterances": 0})["by_marker"]["article"])
        self.assertIsNone(cp.thresholded_summary(counts, {"target_min_marker_utterances": 1, "speaker_min_marker_utterances": 8})["by_marker"]["article"])
        self.assertIsNone(cp.thresholded_summary(counts, {"target_min_marker_utterances": 1, "speaker_min_marker_utterances": 0, "min_exchanges": 11})["by_marker"]["article"])
        exact = cp.thresholded_summary(counts, {"target_min_marker_utterances": 4, "speaker_min_marker_utterances": 7})
        self.assertAlmostEqual(exact["by_marker"]["article"], .3)

    def test_group_aggregation_with_undefined_markers_matches_library(self):
        complete = CoordinationCounts(COORDINATION_MARKERS)
        for _ in range(3):
            complete.add("the and of is very some it we", "the and of is very some it we").add("court", "court")
        partial = CoordinationCounts(COORDINATION_MARKERS).add("the court", "a court").add("court", "the court")  # only article defined
        thresholds = {"target_min_marker_utterances": 1, "speaker_min_marker_utterances": 0}
        group, records = cp.evaluate_group({("x", "case"): complete, ("y", "case"): partial}, thresholds)
        self.assertEqual(group["speakers"], 2)
        self.assertEqual(group["speakers_complete"], 1)
        self.assertAlmostEqual(records[0]["aggregated_1"], records[0]["aggregated_3"])
        self.assertIsNone(records[1]["aggregated_1"])
        self.assertAlmostEqual(records[1]["aggregated_3"], records[1]["by_marker"]["article"])
        # Aggregated 2 fills the partial entity's seven missing markers with the group means (= complete entity's values).
        expected = (records[1]["by_marker"]["article"] + sum(records[0]["by_marker"][m] for m in COORDINATION_MARKERS if m != "article")) / 8
        self.assertAlmostEqual(records[1]["aggregated_2"], expected)
        self.assertAlmostEqual(group["aggregated_2"], (records[0]["aggregated_2"] + records[1]["aggregated_2"]) / 2)
        self.assertAlmostEqual(group["aggregated_3"], (records[0]["aggregated_3"] + records[1]["aggregated_3"]) / 2)

    def test_welch_and_bootstrap_behave(self):
        x, y = np.array([1., 2., 3., 4., 5.]), np.array([1., 1., 2., 2., 1.5])
        result = cp.welch(x, y)
        self.assertGreater(result["t"], 0)
        self.assertLess(result["p"], .2)
        self.assertEqual(cp.welch([1.], [2., 3.])["t"], None)
        self.assertEqual(cp.welch([2., 2.], [2., 2.])["method"], "zero variance")
        low, high = cp.bootstrap_difference(x, y, np.random.default_rng(1), 300)
        self.assertLess(low, x.mean() - y.mean())
        self.assertGreater(high, x.mean() - y.mean())
        self.assertEqual(cp.bootstrap_difference(np.ones(4) * 3, np.ones(4), np.random.default_rng(1), 200), [2., 2.])
        paired = cp.paired_t([1., 1.5, .5, 1.2])
        self.assertEqual(paired["n_pairs"], 4)
        self.assertLess(paired["p"], .05)

    def test_favorability_rules(self):
        meta = {"advocates": {"law": {"side": 1}, "amicus": {"side": 2}}, "votes_side": {"j1": 1, "j2": 0, "j3": -1}}
        self.assertEqual(cp.favorability(meta, "law", "j1"), "favorable")
        self.assertEqual(cp.favorability(meta, "law", "j2"), "unfavorable")
        self.assertEqual(cp.favorability(meta, "law", "j3"), "justice_vote_undefined")
        self.assertEqual(cp.favorability(meta, "amicus", "j1"), "lawyer_side_undefined")
        self.assertEqual(cp.favorability(meta, "nobody", "j1"), "lawyer_side_undefined")
        self.assertEqual(cp.favorability({"advocates": {"law": {"side": 1.0}}, "votes_side": {"j1": 1.0}}, "law", "j1"), "favorable")

    def test_fixture_corpus_end_to_end_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = fixture_corpus(directory)
            summary = cp.run(fixture["config"], directory / "out", fixture["cache"])
            coverage = summary["coverage"]
            self.assertEqual(coverage["matched_exactly"], 1)
            self.assertEqual(coverage["matched_via_alias"], ["01-100b", "01orig"])
            self.assertEqual(coverage["unmatched"], ["77-777"])
            self.assertEqual(coverage["selected_conversations"], 3)
            self.assertEqual(coverage["selected_distinct_case_ids"], 2)
            self.assertEqual(coverage["cases_with_multiple_sessions"], ["2004_01-100"])
            data = summary["data"]
            self.assertEqual(data["dropped"]["no_reply_to"], 3)
            self.assertEqual(data["dropped"]["empty_text"], 6)  # the empty utterance and the reply to it, per session
            self.assertEqual(data["dropped"]["unknown_or_inaudible_speaker"], 6)
            self.assertEqual(data["distinct_justices"], 2)
            self.assertEqual(data["distinct_lawyers"], 2)
            self.assertEqual(data["chief_justice_terms"], {"j__chief": [2004]})
            self.assertEqual(data["justice_entities"], 4)
            self.assertEqual(data["lawyer_entities"], 3)  # lawyer_a in both cases, lawyer_b in 01-100 only
            self.assertEqual(data["exchanges_kept_total"], sum(data["exchanges_kept"].values()))
            self.assertEqual(data["favorability_exchanges"].get("lawyer_speaker:lawyer_side_undefined", 0) > 0, True)  # amicus in 01-orig
            self.assertEqual(summary["votes_cross_check_against_cases_jsonl"], {"advocate_sides_agree": 3, "agree": 2, "disagree": 1})
            groups = summary["primary"]["groups"]
            self.assertEqual(groups["lawyers_to_justices"]["speakers"], 3)
            self.assertEqual(groups["lawyers_to_chief"]["speakers"] + groups["lawyers_to_associates"]["speakers"], 6)
            self.assertEqual(groups["lawyers_to_favorable"]["speakers"], 2)  # entities with side 0/1 in 01-100 only
            self.assertEqual(groups["pair_lawyers_to_favorable"]["speakers"] + groups["pair_lawyers_to_unfavorable"]["speakers"], 4)
            claims = summary["primary"]["claims"]
            self.assertEqual(set(claims), set(cp.CLAIMS) | set(cp.PAIR_CLAIMS))
            self.assertIn("paired_sensitivity", claims["F5a_lawyers_to_unfavorable_over_favorable"])
            self.assertIn("strict", summary["sensitivities"])
            self.assertEqual(summary["sensitivities"]["exact_notebook_case_ids"]["excluded_dockets"], ["01-orig"])
            self.assertEqual(summary["sensitivities"]["exact_notebook_case_ids"]["groups"]["lawyers_to_justices"]["speakers"], 2)
            out = directory / "out"
            for name in ("summary.json", "manifest.json", "config.json", "runner.py", "report.md", "entities.jsonl.gz"):
                self.assertTrue((out / name).exists(), name)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["artifacts"]["summary.json"], hashlib.sha256((out / "summary.json").read_bytes()).hexdigest())
            self.assertFalse(manifest["raw_text_written"])
            texts = {u["text"] for u in fixture["utterances"] if len(u["text"].split()) >= 3}
            with gzip.open(out / "entities.jsonl.gz", "rt") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), manifest["entity_records"])
            self.assertTrue(all(set(row["counts"]) == {"a", "b", "ab"} and "text" not in row for row in rows))
            dumped = (out / "entities.jsonl.gz").read_bytes()
            self.assertFalse(any(text in gzip.decompress(dumped).decode() for text in texts))
            again = cp.run(fixture["config"], directory / "out2", fixture["cache"])
            self.assertEqual((out / "summary.json").read_bytes(), (directory / "out2" / "summary.json").read_bytes())
            self.assertEqual(again["verdicts"], summary["verdicts"])

    def test_corrupted_input_and_nonempty_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = fixture_corpus(directory)
            (fixture["cache"] / "cases.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                cp.run(fixture["config"], directory / "out", fixture["cache"])
            self.assertFalse((directory / "out").exists())
            output = directory / "busy"
            output.mkdir()
            (output / "precious").write_text("keep")
            with self.assertRaises(ValueError):
                cp.run(fixture["config"], output, fixture["cache"])
            self.assertEqual((output / "precious").read_text(), "keep")
            with self.assertRaises(FileNotFoundError):
                cp.run(dict(fixture["config"], inputs={"missing.zip": {"sha256": "0" * 64}}), directory / "out3", fixture["cache"])

    def test_frozen_config_is_consistent(self):
        config = cp.load_config(cp.DEFAULT_CONFIG)
        self.assertEqual(len(config["case_ids"]), 204)
        self.assertEqual(len(set(config["case_ids"])), 204)
        self.assertEqual(config["markers"], list(COORDINATION_MARKERS))
        for alias, target in config["case_id_aliases"].items():
            self.assertIn(alias, config["case_ids"])
            if alias.endswith("b"):
                self.assertIn(target, config["case_ids"])
        for spec in config["inputs"].values():
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(config["statistics"]["bootstrap_samples"], 100)


if __name__ == "__main__":
    unittest.main()
