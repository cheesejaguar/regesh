import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from regesh.bonding.synchrony import COORDINATION_MARKERS, LEXICON, CoordinationCounts
from regesh.providers import ModelResult, ProviderError
from regesh.reproductions.llm_convergence import (aggregated_3, analyze, content_vocabulary, convergence_from_flags,
                                                  feature_flags, load_dialogues, measured_reply, messages,
                                                  normalize_text, parse_eou_lines, resample_aggregates, run, select_items)

DIALOGUES = [
    ["Say , Jim , how about going for a few beers after dinner ?", "You know that is tempting but is really not good for our fitness .",
     "What do you mean ? It will help us to relax .", "Do you really think so ? I don ' t .", "I guess you are right .", "Let ' s go to the gym ."],
    ["Can you do push-ups ?", "Of course I can . It ' s a piece of cake !", "Really ? I think that ' s impossible !", "You mean 30 push-ups ?", "Yeah !"],
    ["Hi", "Hello there , how are you ?", "Fine .", "Good ."],
    ["Where are you going today ?", "I am going to the market .", "Could you buy me some apples ?", "Sure , how many do you want ?",
     "Maybe five or six .", "No problem .", "Thanks a lot .", "You are welcome ."],
    ["This is a long dialogue .", "Yes it is .", "Very long indeed .", "Longer than most .", "Much longer .", "Still going .",
     "Nearly done .", "Almost there .", "One more turn ."],
    ["Too short ."],
    ["Where are you going today ?", "I am going to the market .", "Could you buy me some apples ?", "Sure , how many do you want ?",
     "Maybe five or six .", "No problem .", "Thanks a lot .", "You are welcome ."],
    ["Is the meeting at three today ?", "Yes , the meeting starts at three in the main room .", "Should I bring the report ?",
     "Please bring two copies of the report .", "I will print them now .", "Thank you very much ."],
    ["Did you see the game last night ?", "I missed it , how was it ?", "The home team won in the last minute .", "That must have been exciting !"],
    ["My car would not start this morning .", "Oh no , what did you do ?", "I took the bus to work instead .", "That was a sensible choice .",
     "The mechanic is looking at it now .", "I hope it is nothing serious ."],
]
CONFIG = {'seed': 5, 'turn_range': [4, 8], 'cut_points': [2, 3, 4], 'min_tokens': 3, 'items': 4,
          'conditions': ['plain', 'bounded'], 'system_prompt': 'You are the second speaker. Reply with one line.',
          'bounded_sentence': 'Do not echo the other speaker.'}


def analysis_config(items, **extra):
    base = {'items': items, 'calls': 2 * items, 'conditions': ['plain', 'bounded'], 'markers': list(COORDINATION_MARKERS),
            'shuffle_permutations': 20, 'shuffle_seed': 1, 'bootstrap_samples': 50, 'bootstrap_seed': 2,
            'content_proxy': {'vocabulary_size': 100, 'min_word_length': 4},
            'published_reference': {'liwc_mean': {'wildchat_llm': .068, 'wildchat_user': .035},
                                    'noun_mean': {'wildchat_llm': .442, 'wildchat_user': .141},
                                    'per_marker': {'wildchat_llm': {m: .05 for m in COORDINATION_MARKERS},
                                                   'wildchat_user': {m: .03 for m in COORDINATION_MARKERS}}}}
    base.update(extra)
    return base


def make_rows(u1s, humans, plains, boundeds):
    rows = []
    for i, (u1, h, pl, bd) in enumerate(zip(u1s, humans, plains, boundeds)):
        for condition, reply in (('plain', pl), ('bounded', bd)):
            rows.append({'id': f'd{i}', 'index': i, 'k': 2, 'condition': condition, 'u1': u1, 'u2_human': h, 'reply': reply,
                         'error': None, 'finish_reason': 'stop', 'usage': {'completion_tokens': 3, 'prompt_tokens': 20},
                         'elapsed_seconds': .5})
    return rows


class LLMConvergenceTests(unittest.TestCase):
    def test_normalization_restores_contractions_and_punctuation(self):
        self.assertEqual(normalize_text("Let ' s go , don ' t you ? It ’ s five o ' clock ."), "Let's go, don't you? It's five o'clock.")
        self.assertEqual(normalize_text("  I  am   fine ! "), 'I am fine!')
        self.assertEqual(normalize_text("the boys ' room"), "the boys ' room")
        with self.assertRaises(ValueError):
            normalize_text(None)

    def test_selection_is_deterministic_and_pre_outcome(self):
        items = select_items(DIALOGUES, CONFIG)
        self.assertEqual(items, select_items(list(DIALOGUES), CONFIG))
        self.assertEqual(len(items), 4)
        self.assertEqual(len({i['id'] for i in items}), 4)
        self.assertNotIn('dailydialog-train:6', {i['id'] for i in items})  # exact duplicate of index 3
        for item in items:
            self.assertIn(item['k'], (2, 3, 4))
            self.assertEqual(len(item['context']), item['k'])
            self.assertEqual(item['context'][-1], item['u1'])
            self.assertTrue(4 <= item['turns'] <= 8)
            self.assertNotIn(" ' ", item['u1'])
            self.assertEqual(item['order_sha256'], hashlib.sha256(
                f"5:select:{item['source_index']}:{' __eou__ '.join(normalize_text(t) for t in DIALOGUES[item['source_index']])}".encode()).hexdigest())
        self.assertEqual([i['order_sha256'] for i in items], sorted(i['order_sha256'] for i in items))
        self.assertNotEqual(items, select_items(DIALOGUES, {**CONFIG, 'seed': 6}))
        with self.assertRaises(ValueError):
            select_items(DIALOGUES, {**CONFIG, 'items': 50})

    def test_messages_for_both_conditions(self):
        odd = {'k': 3, 'context': ['Hi there.', 'Hello.', 'How are you?'], 'u1': 'How are you?'}
        even = {'k': 2, 'context': ['Hi there.', 'Hello, how are you?'], 'u1': 'Hello, how are you?'}
        plain = messages(odd, 'plain', CONFIG); bounded = messages(odd, 'bounded', CONFIG)
        self.assertEqual([m['role'] for m in plain], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(plain[-1]['content'], odd['u1'])
        self.assertEqual(plain[1:], bounded[1:])
        self.assertEqual(bounded[0]['content'], CONFIG['system_prompt'] + ' ' + CONFIG['bounded_sentence'])
        self.assertEqual([m['role'] for m in messages(even, 'plain', CONFIG)], ['system', 'assistant', 'user'])
        with self.assertRaises(ValueError):
            messages(odd, 'other', CONFIG)
        with self.assertRaises(ValueError):
            messages({'k': 2, 'context': ['a', 'b'], 'u1': 'c'}, 'plain', CONFIG)

    def test_measured_reply_cleanup(self):
        self.assertEqual(measured_reply('B: "Sure , I don \' t mind ."\nA: next line'), "Sure, I don't mind.")
        self.assertEqual(measured_reply('\n\n  Speaker 2: fine.  '), 'fine.')
        self.assertEqual(measured_reply('whatever', error='HTTPError'), '')
        self.assertEqual(measured_reply('   '), '')
        self.assertEqual(measured_reply(None), '')

    def test_original_zip_layout_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, 'w') as z:
                z.writestr('train/dialogues_train.txt', 'Hi there . __eou__ Hello . __eou__ \nOne . __eou__ Two . __eou__ Three . __eou__\n')
            archive = Path(directory) / 'ijcnlp_dailydialog.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('ijcnlp_dailydialog/train.zip', inner.getvalue())
            dialogues, source = load_dialogues(directory)
            self.assertEqual(dialogues, [['Hi there .', 'Hello .'], ['One .', 'Two .', 'Three .']])
            self.assertEqual(source['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(parse_eou_lines(['']), [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_dialogues(directory)

    def test_convergence_matches_module_on_known_values(self):
        u1 = ['the cat sat', 'cats sit', 'a dog ran', 'dogs run']
        replies = ['the dog', 'dogs', 'a cat', 'cats']          # article iff target has one
        flags_a = feature_flags(u1, [LEXICON[m] for m in COORDINATION_MARKERS])
        flags_b = feature_flags(replies, [LEXICON[m] for m in COORDINATION_MARKERS])
        conv = convergence_from_flags(flags_a, flags_b)
        article = COORDINATION_MARKERS.index('article')
        self.assertAlmostEqual(conv[article], 1. - .5)
        self.assertTrue(np.isnan(conv[COORDINATION_MARKERS.index('ppron')]))
        counts = CoordinationCounts()
        for t, r in zip(u1, replies):
            counts.add(t, r)
        summary = counts.summary()
        self.assertAlmostEqual(summary['by_marker']['article'], .5)
        self.assertIsNone(summary['by_marker']['ppron'])
        self.assertAlmostEqual(float(aggregated_3(conv)), summary['aggregated_3'])
        divergent = convergence_from_flags(feature_flags(u1, [LEXICON['article']]),
                                           feature_flags(['dog', 'the dogs', 'cat', 'the cats'], [LEXICON['article']]))
        self.assertAlmostEqual(divergent[0], -.5)

    def test_resampling_shapes_and_identity(self):
        u1 = ['the cat sat on it', 'cats sit', 'a dog ran', 'dogs run and jump']
        a = feature_flags(u1, [LEXICON[m] for m in COORDINATION_MARKERS])
        b = {'x': feature_flags(['the dog', 'dogs', 'a cat', 'cats and dogs'], [LEXICON[m] for m in COORDINATION_MARKERS])}
        identity = np.tile(np.arange(4), (3, 1))
        out = resample_aggregates(a, b, identity, chunk=2)
        self.assertEqual(out['x'].shape, (3,))
        self.assertTrue(np.allclose(out['x'], aggregated_3(convergence_from_flags(a, b['x']))))
        rng = np.random.default_rng(0)
        boot = resample_aggregates(a, b, rng.integers(0, 4, (25, 4)))
        self.assertEqual(boot['x'].shape, (25,))
        self.assertEqual(content_vocabulary(['apple banana the apple', 'cherry banana'], 2, 4), ['apple', 'banana'])
        self.assertEqual(content_vocabulary(['the and with'], 5, 4), [])

    def test_analysis_on_fixture_with_known_convergence(self):
        u1 = ['the cat sat', 'cats sit', 'a dog ran', 'dogs run']
        humans = ['the dog', 'dogs', 'a cat', 'cats']            # +.5 on article
        plains = ['the dog', 'the dogs', 'a cat', 'a cats']       # always article: 1 - 1 = 0
        boundeds = ['dog', 'dogs', 'cat', 'cats']                 # never: undefined everywhere -> None
        rows = make_rows(u1, humans, plains, boundeds)
        result = analyze(rows, analysis_config(4))
        sets = result['response_sets']
        self.assertAlmostEqual(sets['human']['coordination']['by_marker']['article'], .5)
        self.assertAlmostEqual(sets['human']['coordination']['aggregated_3'], .5)
        self.assertAlmostEqual(sets['plain']['coordination']['aggregated_3'], 0.)
        self.assertAlmostEqual(sets['bounded']['coordination']['aggregated_3'], 0.)  # defined: targets exhibit articles
        self.assertEqual(sets['bounded']['accommodation_assessment']['band'], 'under')
        self.assertEqual(sets['human']['accommodation_assessment']['band'], 'over')
        self.assertEqual(sets['plain']['empty_replies'], 0)
        self.assertEqual(sets['plain']['provider_failures'], 0)
        self.assertEqual(sets['plain']['actual_tokens']['completion_tokens'], 12)
        self.assertAlmostEqual(result['bootstrap']['markers']['plain_minus_human']['point'], -.5)
        ci = result['bootstrap']['markers']['plain_minus_human']['ci95']
        self.assertEqual(len(ci), 2); self.assertLessEqual(ci[0], ci[1])
        self.assertEqual(result['shuffled_baseline']['permutations'], 20)
        self.assertIn('shuffled_95_band', result['shuffled_baseline']['markers']['human'])
        self.assertFalse(result['comparison']['same_direction_as_published'])
        self.assertAlmostEqual(result['comparison']['published_eight_marker_llm_minus_user_derived'], .02)
        self.assertEqual(result['content_vocabulary'], ['cats', 'dogs'])
        with self.assertRaises(ValueError):
            analyze(rows[:-1], analysis_config(4))
        with self.assertRaises(ValueError):
            analyze(rows + [rows[0]], analysis_config(4))

    def test_empty_and_failed_replies_are_counted_not_dropped(self):
        u1 = ['the cat sat', 'cats sit', 'a dog ran', 'dogs run']
        rows = make_rows(u1, ['the dog', 'dogs', 'a cat', 'cats'], ['', 'the dogs', 'a cat', 'cats'], ['x', 'y', 'z', 'w'])
        rows[0]['error'] = 'HTTPError'; rows[0]['finish_reason'] = None
        result = analyze(rows, analysis_config(4))
        self.assertEqual(result['response_sets']['plain']['empty_replies'], 1)
        self.assertEqual(result['response_sets']['plain']['provider_failures'], 1)
        self.assertEqual(result['response_sets']['plain']['finish_reasons']['provider_error'], 1)
        self.assertEqual(result['response_sets']['plain']['n'], 4)
        self.assertIsNotNone(result['response_sets']['plain']['lsm_mean_non_empty'])

    def test_locked_items_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = d / 'config.json'; items = d / 'items.jsonl'; out = d / 'out'
            cfg.write_text(json.dumps({'items_sha256': 'wrong'})); items.write_text('{}\n')
            with self.assertRaises(ValueError):
                run(cfg, items, out, None)

    def test_recorded_end_to_end_with_fixture_model(self):
        class Model:
            model = 'unit-test-fixture'
            calls = 0

            def complete(self, prompt, *, seed, max_tokens):
                Model.calls += 1
                if Model.calls == 3:
                    raise ProviderError('down', {'request': {'messages': prompt, 'seed': seed}, 'error_type': 'URLError',
                                                 'elapsed_seconds': .1})
                text = 'B: ' + prompt[-1]['content']   # verbatim echo of u1
                response = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}
                return ModelResult(text, {'request': {'messages': prompt, 'seed': seed, 'max_tokens': max_tokens, 'temperature': 0},
                                          'response': response, 'raw_response': json.dumps(response), 'elapsed_seconds': .2})
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = d / 'config.json'; items_path = d / 'items.jsonl'; out = d / 'out'
            items = select_items(DIALOGUES, CONFIG)
            payload = ''.join(json.dumps(i) + '\n' for i in items)
            items_path.write_text(payload)
            config = analysis_config(4, **{k: v for k, v in CONFIG.items() if k != 'items'},
                                     items_sha256=hashlib.sha256(payload.encode()).hexdigest(), max_tokens=8, temperature=0)
            cfg.write_text(json.dumps(config))
            summary = run(cfg, items_path, out, Model())
            self.assertEqual(Model.calls, 8)
            rows = [json.loads(l) for l in (out / 'rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual([r['condition'] for r in rows[:4]], ['plain', 'bounded', 'bounded', 'plain'])
            self.assertEqual(sum(bool(r['error']) for r in rows), 1)
            self.assertEqual([r['reply'] for r in rows if r['error']], [''])
            self.assertEqual({r['index'] + config['seed'] for r in rows}, {json.loads(l)['record']['request']['seed'] for l in (out / 'calls.jsonl').read_text().splitlines()})
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(manifest['complete']); self.assertEqual(manifest['model_calls_completed'], 8)
            self.assertEqual(manifest['max_completion_tokens'], 64)
            self.assertIn('rows.jsonl', manifest['artifact_sha256'])
            echoed = summary['response_sets']['plain']
            self.assertEqual(echoed['empty_replies'], 0)
            self.assertEqual(summary['response_sets']['bounded']['empty_replies'], 1)
            # Verbatim echo: P(b|a) = 1 and P(b) = P(a), so Eq. 1 gives exactly 1 - P(a) per defined marker.
            u1 = [normalize_text(i['u1']) for i in items]
            a = feature_flags(u1, [LEXICON[m] for m in COORDINATION_MARKERS])
            for j, m in enumerate(COORDINATION_MARKERS):
                value = echoed['coordination']['by_marker'][m]
                if a[j].sum() == 0:
                    self.assertIsNone(value)
                else:
                    self.assertAlmostEqual(value, 1 - a[j].sum() / len(u1))
                    self.assertGreaterEqual(value, 0)
            with self.assertRaises(ValueError):
                run(cfg, items_path, out, Model())
            with self.assertRaises(ValueError):
                run(cfg, items_path, out, Model(), resume=True)   # complete runs are never extended

    def test_interrupted_run_resumes_without_duplicates(self):
        class Crashing:
            model = 'unit-test-fixture'
            calls = 0
            crash_at = 3

            def complete(self, prompt, *, seed, max_tokens):
                Crashing.calls += 1
                if Crashing.calls == Crashing.crash_at:
                    raise KeyboardInterrupt
                response = {'choices': [{'message': {'content': 'Sure.'}, 'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
                return ModelResult('Sure.', {'request': {'messages': prompt, 'seed': seed, 'max_tokens': max_tokens, 'temperature': 0},
                                             'response': response, 'raw_response': json.dumps(response), 'elapsed_seconds': .1})
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = d / 'config.json'; items_path = d / 'items.jsonl'; out = d / 'out'
            items = select_items(DIALOGUES, CONFIG)
            payload = ''.join(json.dumps(i) + '\n' for i in items)
            items_path.write_text(payload)
            config = analysis_config(4, **{k: v for k, v in CONFIG.items() if k != 'items'},
                                     items_sha256=hashlib.sha256(payload.encode()).hexdigest(), max_tokens=8, temperature=0)
            cfg.write_text(json.dumps(config))
            with self.assertRaises(KeyboardInterrupt):
                run(cfg, items_path, out, Crashing())
            self.assertEqual(len((out / 'rows.jsonl').read_text().splitlines()), 2)
            with (out / 'calls.jsonl').open('a') as handle:
                handle.write('{"id": "truncated"')        # interrupted partial write
            with self.assertRaises(ValueError):
                run(cfg, items_path, out, Crashing())      # non-empty output without --resume
            Crashing.crash_at = -1
            summary = run(cfg, items_path, out, Crashing(), resume=True)
            rows = [json.loads(l) for l in (out / 'rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({(r['id'], r['condition']) for r in rows}), 8)
            self.assertEqual(len((out / 'calls.jsonl').read_text().splitlines()), 8)
            self.assertEqual(Crashing.calls, 3 + 6)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(manifest['complete']); self.assertEqual(len(manifest['resumed_utc']), 1)
            self.assertEqual(summary['response_sets']['plain']['provider_failures'], 0)


if __name__ == '__main__':
    unittest.main()
