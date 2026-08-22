#!/usr/bin/env python3
"""
Zimi MCP Server — streamable-HTTP transport (single entry point).

`zimi.mcp_server` builds the shared FastMCP instance at import time; this
module decides *how to run it*:

  stdio  (default)  `python -m zimi.mcp_server`
  http            `python -m zimi.mcp_server --http`   (or ZIMI_MCP_TRANSPORT=http)

The streamable-HTTP endpoint lets an AI client reach Zimi over a URL
(`http://host:port/mcp`) instead of spawning a subprocess. This is what
remote/shared deployments and clients that prefer a URL over a command use.

Why a small launcher here instead of `mcp.run(transport="streamable-http")`?
The MCP Python SDK moved the streamable-HTTP config between majors — the repo
supports both 1.x and 2.x:

  1.x (mcp.server.fastmcp.FastMCP)
      host/port/streamable_http_path/stateless_http live in the constructor
      and are exposed as a mutable `settings` object; `streamable_http_app()`
      takes no arguments (it reads `settings`).

  2.x (mcp.server.mcpserver.MCPServer)
      the constructor takes none of them; `streamable_http_app()` takes
      `streamable_http_path`, `stateless_http`, `host`, ...; `run()` takes
      host/port.

So there is no single `run(...)` call that is valid on both. This launcher
builds the Starlette app via `streamable_http_app()` (version-branching only
the *arguments*) and runs it under uvicorn ourselves. uvicorn, starlette, and
sse-starlette are already pulled in transitively by the `mcp` package, so no
new dependency is added.

Configuration (env vars / CLI):
  ZIM_DIR           Path to *.zim files (default: /zims)          [also stdio]
  ZIMI_MCP_TRANSPORT  "http" (or set --http) enables this transport
  ZIMI_MCP_HOST       Bind address          (default: 0.0.0.0)
  ZIMI_MCP_PORT       Bind port             (default: 8100)
  ZIMI_MCP_PATH       MCP endpoint path     (default: /mcp)
"""

import argparse
import os
import sys


def _mcp_major() -> int:
    """Best-effort major version of the installed MCP SDK (1 or 2+).

    The transport-config location differs between these; everything else is
    shared. If we can't tell, assume 1.x (the branch that needs no args).
    """
    try:
        from importlib.metadata import version

        return int(version("mcp").split(".")[0])
    except Exception:
        return 1


def _stateless_default() -> bool:
    # Stateless (serverless) HTTP: each request is independent, no persistent
    # session. Correct default for a shared/reverse-proxied deployment — the
    # MCP initialize/list/call sequence works without a server-side session.
    return os.environ.get("ZIMI_MCP_STATELESS", "1") not in ("0", "false", "False")


def build_http_server(server, path: str, stateless: bool) -> None:
    """Make `server`'s transport *settings* match the streamable HTTP we'll run.

    On 1.x, `streamable_http_app()` reads from `server.settings`; we set those
    fields. On 2.x there is nothing to configure — the values are passed to
    `streamable_http_app()` at build time in `serve_streamable_http()`.
    """
    if _mcp_major() >= 2:
        return
    settings = getattr(server, "settings", None)
    if settings is None:  # pragma: no cover - defensive
        return
    try:
        settings.streamable_http_path = path
        settings.stateless_http = stateless
        settings.json_response = False
    except Exception:  # pragma: no cover - defensive
        pass


def serve_streamable_http(server, host: str, port: int) -> None:
    """Run `server` over streamable HTTP on host:port (blocking).

    Builds the Starlette app (version-aware) and drives uvicorn so the same
    code path works on MCP SDK 1.x and 2.x.
    """
    import uvicorn

    path = os.environ.get("ZIMI_MCP_PATH", "/mcp")
    if not path.startswith("/"):
        path = "/" + path
    os.environ["ZIMI_MCP_PATH"] = path
    stateless = _stateless_default()

    if _mcp_major() >= 2:
        app = server.streamable_http_app(
            streamable_http_path=path,
            stateless_http=stateless,
            json_response=False,
            host=host,
        )
    else:
        # 1.x: streamable_http_app() reads `settings` (configured above).
        app = server.streamable_http_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("ZIMI_MCP_LOG_LEVEL", "info").lower(),
    )


def run_main(server) -> None:
    """Single entry point: pick the transport and run it.

    stdio (default, unchanged behaviour) or streamable HTTP (--http /
    ZIMI_MCP_TRANSPORT=http).
    """
    argv = sys.argv[1:]
    use_http = ("--http" in argv) or (
        os.environ.get("ZIMI_MCP_TRANSPORT", "stdio") == "http"
    )

    if not use_http:
        server.run(transport="stdio")
        return

    parser = argparse.ArgumentParser(description="Zimi MCP server (streamable HTTP)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run over streamable HTTP instead of stdio (same as ZIMI_MCP_TRANSPORT=http).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("ZIMI_MCP_HOST", "0.0.0.0"),
        help="Bind address (default: ZIMI_MCP_HOST or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ZIMI_MCP_PORT", "8100")),
        help="Bind port (default: ZIMI_MCP_PORT or 8100).",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("ZIMI_MCP_PATH", "/mcp"),
        help="MCP endpoint path (default: /mcp).",
    )
    args = parser.parse_args(argv)

    path = args.path if args.path.startswith("/") else "/" + args.path
    os.environ["ZIMI_MCP_PATH"] = path
    build_http_server(server, path, _stateless_default())

    print(f"Zimi MCP streamable HTTP: http://{args.host}:{args.port}{path}", flush=True)
    serve_streamable_http(server, args.host, args.port)
