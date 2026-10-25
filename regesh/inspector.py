"""Small loopback inspector for evidence, trajectories, controls and interventions."""
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import secrets
from urllib.parse import parse_qs, urlsplit
from .agent import Agent
from .schema import canonical


def render(agent, token=""):
    state = agent.get_state()
    trace = agent.trace()
    rows = []
    points = {"valence": [], "arousal": []}
    for i, item in enumerate(trace):
        s = item["state"]
        x = 20 + 700 * i / max(1, len(trace) - 1)
        for key in points:
            points[key].append(f"{x:.2f},{100 - 75*s['mood'][key]:.2f}")
        rows.append(f"<details><summary>{item['seq']} · {escape(item['operation'])} · simulation time {s['time']:.2f}</summary>"
                    f"<h4>Trigger / recorded result</h4><pre>{escape(json.dumps(item['payload'],indent=2))}</pre>"
                    f"<h4>State after transition</h4><pre>{escape(json.dumps(s,indent=2))}</pre></details>")
    svg = ''.join(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points[key])}"/>'
                  for key, color in (("valence", "#1455aa"), ("arousal", "#ba5100")))
    # Escaping '<' prevents source events from closing the script tag.
    token_json = json.dumps(token).replace("<", "\\u003c")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Regesh inspector</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:2em auto;padding:0 1em;color:#172333}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:1em}}details{{border-bottom:1px solid #ccd;padding:.7em}}textarea{{width:95%;min-height:5em}}button{{padding:.5em 1em}}svg{{width:100%;border:1px solid #ddd}}label{{display:block;margin-top:1em}}</style></head><body>
<h1>Regesh · {escape(agent.id)}</h1>
<p>Functional affect research. Blue: valence; orange: arousal. X: journal order. Y: state value (−1 to 1). No claim of subjective experience.</p>
<svg viewBox="0 0 740 200" role="img" aria-label="Mood trajectory"><line x1="0" x2="740" y1="100" y2="100" stroke="#ccc"/>{svg}</svg>
<h2>Current state and cognitive controls</h2><pre>{escape(json.dumps({'time':state['time'],'mood':state['mood'],'internal':state['internal'],'controls':agent.actuation_context()},indent=2))}</pre>
<p><a href="/api/state">Full state JSON</a> · <a href="/api/trace">Journal JSON</a> · <a href="/api/snapshot">Snapshot JSON</a></p>
<h2>Research interventions</h2><p>Interventions are journaled separately from observed evidence. Fork before changing state to compare trajectories.</p>
<label>Fork or replay target agent ID <input id="target" value="comparison"></label>
<button onclick="send('fork',{{agent_id:document.getElementById('target').value}})">Fork current snapshot</button>
<label>Replay through journal sequence (blank = latest) <input id="through" type="number" min="0"></label>
<button onclick="send('replay',{{agent_id:document.getElementById('target').value,through:document.getElementById('through').value===''?null:Number(document.getElementById('through').value)}})">Replay to target</button>
<label>State changes (JSON)<textarea id="changes">{{"mood":{{"valence":0.4}}}}</textarea></label>
<label>Intervention reason <input id="reason" value="controlled comparison"></label>
<button onclick="send('intervene',{{changes:JSON.parse(document.getElementById('changes').value),reason:document.getElementById('reason').value}})">Apply intervention</button>
<label>Advance simulation time <input id="dt" type="number" min="0" value="5"></label>
<button onclick="send('advance',{{dt:Number(document.getElementById('dt').value)}})">Advance</button>
<pre id="result"></pre><h2>Causal journal</h2>{''.join(rows)}
<script>const token={token_json};async function send(op,payload){{try{{const r=await fetch('/api/'+op,{{method:'POST',headers:{{'Content-Type':'application/json','X-Regesh-Token':token}},body:JSON.stringify(payload)}});document.getElementById('result').textContent=await r.text();if(r.ok&&op==='intervene'||r.ok&&op==='advance')location.reload();}}catch(e){{document.getElementById('result').textContent=String(e)}}}}</script></body></html>'''


def make_server(db_path, agent_id="x", host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Inspector binds to loopback only")
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, status, value, content_type="application/json"):
            body = value.encode() if isinstance(value, str) else canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _local_host(self):
            hostname = urlsplit("//" + self.headers.get("Host", "")).hostname
            return hostname in {"127.0.0.1", "localhost", "::1"}

        def do_GET(self):
            if not self._local_host():
                return self._send(403, {"error": "loopback host required"})
            agent = Agent(db_path, agent_id)
            try:
                path = urlsplit(self.path).path
                if path == "/":
                    self._send(200, render(agent, token), "text/html")
                elif path == "/api/state":
                    self._send(200, agent.get_state())
                elif path == "/api/trace":
                    self._send(200, agent.trace())
                elif path == "/api/snapshot":
                    self._send(200, agent.snapshot())
                else:
                    self._send(404, {"error": "unknown operation"})
            finally:
                agent.close()

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected_origin = "http://" + self.headers.get("Host", "")
            if not self._local_host() or self.headers.get("X-Regesh-Token") != token or origin and origin != expected_origin:
                return self._send(403, {"error": "local inspector token/origin required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024*1024:
                    raise ValueError("request size outside limit")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON content type required")
                payload = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                agent = Agent(db_path, agent_id)
                try:
                    if path == "/api/intervene":
                        result = agent.intervene(payload["changes"], payload["reason"])
                    elif path == "/api/advance":
                        result = {"simulation_time": agent.advance_time(payload["dt"])}
                    elif path in {"/api/fork", "/api/replay"}:
                        child = agent.fork(payload["agent_id"]) if path.endswith("fork") else agent.replay(payload["agent_id"], through=payload.get("through"))
                        try:
                            result = {"agent_id": child.id, "snapshot_digest": child.snapshot()["digest"]}
                        finally:
                            child.close()
                    else:
                        return self._send(404, {"error": "unknown operation"})
                    self._send(200, result)
                finally:
                    agent.close()
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})

    return HTTPServer((host, port), Handler)


def serve(db_path, agent_id="x", host="127.0.0.1", port=8765):
    server = make_server(db_path, agent_id, host, port)
    print(f"Regesh inspector: http://{host}:{server.server_port} (agent {agent_id})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
