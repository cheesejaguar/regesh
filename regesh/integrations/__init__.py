"""Integrations that attach the affect loop to existing agent harnesses.

Three surfaces are used, each optional and reversible:

* an MCP stdio server (`regesh-mcp`) exposing affect tools and a context resource;
* harness hooks (`regesh-hook`) that observe prompts, tool results and turn ends and
  inject a short factual affect context where the harness allows it;
* an instruction block appended to the harness's global instruction file.

All state lives under ~/.regesh. Nothing here grants permissions, calls the network
(unless the host configures the embedding sensor) or blocks a harness action: hooks
fail open and exit 0.
"""
