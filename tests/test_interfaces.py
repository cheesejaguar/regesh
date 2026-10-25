"""Integration checks against a real local HTTP server, no external model calls."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from regesh import Agent, Event
from regesh.providers import FixtureModel, LocalChatModel, ModelAppraisal, ProviderError
from regesh.inspector import make_server, render
from regesh.cli import demo


class FakeModelServer:
    def __init__(self, content="candidate", raw=None):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append({"body": json.loads(body), "path": self.path,
                                       "authorization": self.headers.get("Authorization")})
                response = raw if raw is not None else json.dumps({"id": "fake-call-1", "model": "fake-model",
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "agent.sqlite")
        self.agent = Agent(self.path, "demo")

    def tearDown(self):
        self.agent.close()
        self.temp.cleanup()

    def test_fixture_deterministic_and_journaled(self):
        model = FixtureModel()
        messages = [{"role": "user", "content": "candidate"}]
        first = model.complete(messages, recorder=self.agent.record_artifact)
        second = model.complete(messages)
        self.assertEqual(first, second)
        self.assertIn("fixture", first.record["mode"])
        payload = self.agent.trace()[-1]["payload"]
        self.assertIn("model_call", json.dumps(payload))
        self.assertIn("raw_response", json.dumps(payload))

    def test_compatible_adapter_exact_request_response_and_no_auth_log(self):
        server = FakeModelServer()
        try:
            model = LocalChatModel(server.url, "fake-model", "secret-for-test")
            result = model.complete([{"role": "user", "content": "test"}], seed=17,
                                    max_tokens=19, recorder=self.agent.record_artifact)
            self.assertEqual(result.text, "candidate")
            self.assertEqual(server.requests[0]["path"], "/v1/chat/completions")
            self.assertEqual(server.requests[0]["authorization"], "Bearer secret-for-test")
            self.assertEqual(server.requests[0]["body"]["seed"], 17)
            self.assertEqual(server.requests[0]["body"], result.record["request"])
            self.assertEqual(json.loads(result.record["raw_response"]), result.record["response"])
            self.assertNotIn("secret-for-test", json.dumps(self.agent.trace()))
        finally:
            server.close()

    def test_malformed_response_retained_as_error_artifact(self):
        server = FakeModelServer(raw=b'{"missing":"choices"}')
        try:
            model = LocalChatModel(server.url, "fake")
            with self.assertRaises(ProviderError) as caught:
                model.complete([], recorder=self.agent.record_artifact)
            self.assertEqual(caught.exception.record["raw_response"], '{"missing":"choices"}')
            self.assertEqual(caught.exception.record["error_type"], "KeyError")
            self.assertIn("error_type", json.dumps(self.agent.trace()[-1]["payload"]))
        finally:
            server.close()

    def test_invalid_appraisal_does_not_become_evidence(self):
        server = FakeModelServer(content=json.dumps({"gain": 5}))
        try:
            provider = ModelAppraisal(LocalChatModel(server.url, "fake"))
            other = Agent(self.path, "model", appraisal_provider=provider)
            try:
                before = other.snapshot()["digest"]
                event = Event("new", "test", "person", "goal", "task", "2026-10-25T00:00:00Z")
                with self.assertRaises(ProviderError) as caught:
                    other.observe(event)
                self.assertEqual(before, other.snapshot()["digest"])
                self.assertIn("raw_response", caught.exception.record)
                other.record_artifact("rejected_model_appraisal", caught.exception.record)
                self.assertNotIn("new", other.get_state()["events"])
            finally:
                other.close()
        finally:
            server.close()

    def test_valid_appraisal_records_exact_model_trace(self):
        proposal = {"goal_relevance": .5, "gain": .4, "expected_quality": 0.,
            "surprise": .2, "control": .6, "responsibility": 0., "uncertainty": .4,
            "approval": 0., "support": ["new"], "alternatives": ["intent unclear"],
            "social_meaning": "unspecified"}
        server = FakeModelServer(content=json.dumps(proposal))
        try:
            other = Agent(self.path, "model", appraisal_provider=ModelAppraisal(LocalChatModel(server.url, "fake")))
            try:
                event = Event("new", "test", "person", "goal", "task", "2026-10-25T00:00:00Z")
                result = other.observe(event)
                self.assertEqual(result["gain"], .4)
                self.assertEqual(result["model_record"]["response"]["id"], "fake-call-1")
                self.assertEqual(other.get_state()["events"]["new"]["data"], {})
                replayed = other.replay("model-replay")
                try:
                    self.assertEqual(replayed.get_state(), other.get_state())
                    self.assertEqual(len(server.requests), 1)
                finally:
                    replayed.close()
            finally:
                other.close()
        finally:
            server.close()

    def test_model_appraisal_rejects_invented_evidence(self):
        proposal = {"goal_relevance": .5, "gain": .4, "expected_quality": 0.,
            "surprise": .2, "control": .6, "responsibility": 0., "uncertainty": .4,
            "approval": 0., "support": ["invented"], "alternatives": ["intent unclear"],
            "social_meaning": "unspecified"}
        server = FakeModelServer(content=json.dumps(proposal))
        try:
            provider = ModelAppraisal(LocalChatModel(server.url, "fake"))
            with self.assertRaises(ProviderError):
                provider.appraise(Event("new", "test", "person", "goal", "task", "2026-10-25T00:00:00Z"), self.agent.get_state())
        finally:
            server.close()

    def test_inspector_escapes_observations(self):
        self.agent.observe(Event("html", "test", "person", "goal", "task", "2026-10-25T00:00:00Z",
            data={"text": "<script>alert('bad')</script>"}))
        page = render(self.agent)
        self.assertNotIn("<script>alert('bad')</script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("Causal journal", page)

    def test_inspector_rejects_nonlocal_bind(self):
        with self.assertRaises(ValueError):
            make_server(self.path, host="0.0.0.0")

    def test_inspector_http_read_and_mutation_protection(self):
        server = make_server(self.path, "demo", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/api/state") as response:
                self.assertEqual(json.loads(response.read())["schema_version"], 1)
            request = urllib.request.Request(base + "/api/intervene", json.dumps({"changes":{"mood":{"valence":1}},"reason":"test"}).encode(), {"Content-Type":"application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            self.assertEqual(self.agent.get_state()["mood"]["valence"], 0)
            # The same-origin inspector can obtain its per-server token.
            with urllib.request.urlopen(base) as response:
                page = response.read().decode()
            token = json.loads(page.split("const token=", 1)[1].split(";async", 1)[0])
            request.add_header("X-Regesh-Token", token)
            request.add_header("Origin", base)
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.agent.get_state()["mood"]["valence"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_demo_restart_and_reappraisal(self):
        result = demo(self.path, "demo")
        self.assertTrue(result["persisted_across_restart"])
        self.assertEqual(result["appraisal_versions"], 2)
        self.assertEqual(demo(self.path, "demo")["snapshot_digest"], result["snapshot_digest"])


if __name__ == "__main__":
    unittest.main()
