"""Deterministic fixtures and an explicit local OpenAI-compatible HTTP adapter.

No endpoint is guessed and no remote request is made during construction. The
adapter does not create credentials. Each result includes an exact audit record;
pass Agent.record_artifact as recorder when invoking it outside observe().
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from .schema import canonical, finite
from .appraisal import validate_appraisal


@dataclass
class ModelResult:
    text: str
    record: dict


class ProviderError(RuntimeError):
    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record or {}


class FixtureModel:
    identifier = "fixture-model-v1"

    def complete(self, messages, *, seed=0, max_tokens=512, recorder=None):
        request = {"model": self.identifier, "messages": messages, "seed": seed,
                   "max_tokens": max_tokens, "temperature": 0}
        fingerprint = hashlib.sha256(canonical(request).encode()).hexdigest()[:12]
        text = ("Fixture suggestion: separate candidate generation from executable "
                "constraint checking; compare a baseline and a revised candidate. "
                f"Request {fingerprint}.")
        response = {"choices": [{"message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": None, "completion_tokens": None}}
        record = {"provider": self.identifier, "request": request, "response": response,
                  "raw_response": canonical(response), "elapsed_seconds": 0,
                  "mode": "deterministic fixture; no live model"}
        if recorder:
            recorder("model_call", record)
        return ModelResult(text, record)


class LocalChatModel:
    """Chat completions against a user-configured compatible server (e.g. vLLM).

    base_url must end at the API prefix, commonly http://localhost:8000/v1.
    Auth values are never persisted in records. The JSON request and response,
    raw response text, timing, usage, seed, and model identity are persisted.
    """
    identifier = "local-openai-compatible-v1"

    def __init__(self, base_url=None, model=None, api_key=None, timeout=60, temperature=0):
        self.base_url = base_url or os.environ.get("REGESH_MODEL_BASE_URL")
        self.model = model or os.environ.get("REGESH_MODEL_NAME")
        self.api_key = api_key if api_key is not None else os.environ.get("REGESH_MODEL_API_KEY")
        if not self.base_url or not self.model:
            raise ValueError("Configure REGESH_MODEL_BASE_URL and REGESH_MODEL_NAME, or pass both explicitly")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) API prefix without credentials, query, or fragment")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.temperature = finite(temperature,0,2)

    def complete(self, messages, *, seed=0, max_tokens=512, recorder=None):
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        request = {"model": self.model, "messages": messages, "seed": seed,
                   "temperature": self.temperature, "max_tokens": max_tokens}
        # Canonical encoding rejects non-finite JSON before any network request.
        body = canonical(request).encode()
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        record = {"provider": self.identifier, "endpoint": url, "request": request,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "mode": "live re-execution", "timeout_seconds": self.timeout}
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                record["http_status"] = response.status
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("model response exceeded 4 MiB limit")
                record["raw_response"] = raw.decode("utf-8")
                parsed = json.loads(record["raw_response"])
                canonical(parsed)  # Reject non-finite JSON anywhere in the response.
                record["response"] = parsed
                text = parsed["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("model response content must be text")
        except Exception as exc:
            # Do not stringify urllib errors: URLs may contain sensitive server
            # diagnostics. Exact successful response bodies are recorded above.
            record["elapsed_seconds"] = time.perf_counter() - start
            record["error_type"] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                record["http_status"] = exc.code
                record["raw_response"] = exc.read(4 * 1024 * 1024).decode("utf-8", errors="replace")
                try:
                    record["response"] = json.loads(record["raw_response"])
                except ValueError:
                    pass
            if recorder:
                recorder("model_call", record)
            raise ProviderError("Model call failed; inspect the recorded error type/status", record) from None
        record["elapsed_seconds"] = time.perf_counter() - start
        if recorder:
            recorder("model_call", record)
        return ModelResult(text, record)


class ModelAppraisal:
    """Optional model-proposed interpretation, with validated evidence references.

    Appraisal never changes recorded observations or supplies measured outcomes.
    Its full request/response is embedded in the appraisal so Agent.observe can
    atomically persist it with the event. Invalid proposals raise ProviderError;
    callers can save error.record using Agent.record_artifact after rollback.
    """
    identifier = "model-appraisal-v1"

    def __init__(self, model, seed=0, max_tokens=1024):
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens

    def appraise(self, event, state):
        prompt = {
            "event": event.to_dict(), "goals": state["goals"],
            "expectations": state["expectations"], "internal": state["internal"],
            "allowed_evidence_ids": [event.id, *state["events"].keys()],
        }
        messages = [{"role": "system", "content": (
            "Return one JSON object only. Propose an interpretation, never invent an observed "
            "outcome or infer hostile intent as fact. Required numeric fields: goal_relevance, "
            "surprise, control, responsibility, uncertainty in [0,1]; gain, expected_quality, "
            "approval in [-1,1]. Also include support (list of allowed evidence IDs), "
            "alternatives (nonempty list of plausible interpretations), social_meaning "
            "(string). Preserve ambiguity. Quotation, recall, imagination, prediction, "
            "and reflection are not new external evidence.")},
            {"role": "user", "content": canonical(prompt)}]
        result = self.model.complete(messages, seed=self.seed, max_tokens=self.max_tokens)
        try:
            value = json.loads(result.text)
            validate_appraisal(value)
            if not value["support"] or any(k not in prompt["allowed_evidence_ids"] for k in value["support"]):
                raise ValueError("unsupported evidence reference")
            if not value["alternatives"] or not all(isinstance(a, str) for a in value["alternatives"]):
                raise ValueError("missing alternative interpretations")
            if not isinstance(value.get("social_meaning"), str):
                raise ValueError("missing social meaning")
            value.update(schema_version=1, provider=self.identifier, model_record=result.record)
            canonical(value)
            return value
        except (ValueError, TypeError, KeyError) as exc:
            record = {**result.record, "appraisal_validation_error": type(exc).__name__}
            raise ProviderError("Model appraisal failed schema/evidence validation", record) from None
