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
  ZIMI_MCP_LOG_LEVEL  Log level for the HTTP endpoint (default: warning)
                    (debug/info show per-request lines — uvicorn access log,
                    and MCP-SDK session churn like "Terminating session: None"
                    which is normal in stateless mode; error silences it all).
  ZIMI_MCP_API_KEY  Bearer key the client must send as
                    `Authorization: Bearer <key>` (default: empty = no auth,
                    endpoint open). Empty is the no-op default so existing
                    deployments keep working unchanged.
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


def _mcp_api_key() -> str:
    """Bearer key required on the HTTP endpoint.

    Empty string (the default) means no authentication — the endpoint is open,
    matching historical behaviour. A non-empty key is compared in constant time
    against the ``Authorization: Bearer <key>`` header on every request.
    """
    return os.environ.get("ZIMI_MCP_API_KEY", "").strip()


def _bearer_from_request(scope) -> str:
    """Extract the Bearer token from an ASGI ``scope``'s headers ("" if absent)."""
    for key, value in scope.get("headers") or []:
        if key.lower() == b"authorization":
            try:
                v = value.decode("latin-1").strip()
            except (AttributeError, UnicodeError):
                continue
            # "bearer <token>" / "Bearer<token>" (token = anything up to a space)
            if v.lower().startswith("bearer"):
                rest = v[len("bearer"):].strip()
                if rest:
                    return rest.split(None, 1)[0]
            return ""
    return ""


def wrap_with_auth(app):
    """Return ``app`` wrapped in a Bearer-auth middleware, or ``app`` unchanged.

    A thin ASGI wrapper (not a Starlette middleware) so it wraps whatever the
    SDK returns — a Starlette/Starlite app in either case — and needs no
    import. With no key set it returns the app as-is, so the common
    no-auth path adds zero overhead. When a key is set, every request must
    carry ``Authorization: Bearer <key>``; anything else gets ``401`` with
    ``WWW-Authenticate: Bearer`` (MCP clients then present the key).
    """
    key = _mcp_api_key()
    if not key:
        return app
    key_bytes = key.encode("latin-1")

    async def authed(scope, receive, send):
        if scope.get("type", "") not in ("http", "websocket"):
            return await app(scope, receive, send)
        if _constant_time_equal(_bearer_from_request(scope), key_bytes):
            return await app(scope, receive, send)
        await _send_401(receive, send)

    return authed


def _constant_time_equal(a: str, b: bytes) -> bool:
    """Constant-time string-vs-bytes comparison (no early-exit on length)."""
    import hmac

    return hmac.compare_digest(a.encode("latin-1"), b)


async def _send_401(receive, send) -> None:
    body = b'{"error":"unauthorized"}'
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"www-authenticate", b'Bearer realm="mcp"'),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


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


def _resolve_path() -> str:
    path = os.environ.get("ZIMI_MCP_PATH", "/mcp")
    if not path.startswith("/"):
        path = "/" + path
    os.environ["ZIMI_MCP_PATH"] = path
    return path


def _build_app(server, host: str, path: str, stateless: bool):
    """Build the Starlette app for streamable HTTP (version-aware)."""
    if _mcp_major() >= 2:
        return server.streamable_http_app(
            streamable_http_path=path,
            stateless_http=stateless,
            json_response=False,
            host=host,
        )
    # 1.x: streamable_http_app() reads `settings` (set by build_http_server).
    return server.streamable_http_app()


def _probe_bind(host: str, port: int):
    """Probe-bind the port and report the real bound address.

    getaddrinfo resolves host (IPv4/IPv6/name), bind fails fast if the port
    is busy, and getsockname gives the actual address. We close before
    uvicorn binds its own socket, so this only verifies availability — the
    tiny close→bind race is the same tradeoff the web server makes.
    """
    import socket

    family, _, _, _, addr = socket.getaddrinfo(
        host, int(port), socket.AF_UNSPEC, socket.SOCK_STREAM
    )[0]
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(addr)
        return sock.getsockname()[:2]
    finally:
        sock.close()


def _uvicorn_log_level() -> str:
    # "warning" (not uvicorn's default "info") so per-request access lines —
    # e.g. "HEAD /mcp 405" from client liveness probes — stay quiet unless
    # explicitly asked for.
    return os.environ.get("ZIMI_MCP_LOG_LEVEL", "warning").lower()


def _apply_log_level() -> None:
    """Set the log level for the HTTP endpoint's noisy loggers.

    uvicorn's access log is covered by uvicorn.Config(log_level=...); this
    handles the other spam source: the MCP SDK logs per-connection churn at
    INFO (its session manager start/stop and "Terminating session: None" on
    every request in stateless mode). In-process those lines bubble up to
    the web server's root logger, so we pin the SDK's own logger:
    default "warning" silences INFO; "debug"/"info" re-enable it.
    """
    import logging

    level = getattr(logging, _uvicorn_log_level().upper(), logging.WARNING)
    logging.getLogger("mcp").setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)


def serve_streamable_http(server, host: str, port: int) -> None:
    """Run `server` over streamable HTTP on host:port (blocking).

    Builds the Starlette app (version-aware) and drives uvicorn so the same
    code path works on MCP SDK 1.x and 2.x. Prints a startup line, then a
    "ready" line once the port is verified (mirrors the web server's
    `ZIM Reader API starting on port` / `READY <port>` pair in server.py).
    """
    import uvicorn

    path = _resolve_path()
    stateless = _stateless_default()
    _apply_log_level()
    print(f"MCP HTTP API starting on port {port}", flush=True)
    build_http_server(server, path, stateless)
    app = wrap_with_auth(_build_app(server, host, path, stateless))
    actual_host, actual_port = _probe_bind(host, port)
    if _mcp_api_key():
        print("MCP HTTP bearer auth enabled (ZIMI_MCP_API_KEY)", flush=True)
    print(f"MCP HTTP endpoint served on {actual_host}:{actual_port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=_uvicorn_log_level())


def start_mcp_http_thread(server, host: str, port: int):
    """Start streamable HTTP on a background (daemon) thread (non-blocking).

    In-process counterpart to serve_streamable_http() for running the MCP
    endpoint inside an already-running process (the web server): one process
    then serves the web UI, the BitTorrent engine, and the MCP endpoint
    together — no second container needed. Returns
    (thread, actual_host, actual_port).
    """
    import threading
    import uvicorn

    path = _resolve_path()
    stateless = _stateless_default()
    _apply_log_level()
    print(f"MCP HTTP API starting on port {port}", flush=True)
    build_http_server(server, path, stateless)
    app = wrap_with_auth(_build_app(server, host, path, stateless))
    actual_host, actual_port = _probe_bind(host, port)
    if _mcp_api_key():
        print("MCP HTTP bearer auth enabled (ZIMI_MCP_API_KEY)", flush=True)
    print(f"MCP HTTP endpoint served on {actual_host}:{actual_port}", flush=True)
    config = uvicorn.Config(app, host=host, port=port, log_level=_uvicorn_log_level())
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, daemon=True, name="mcp-http")
    thread.start()
    return thread, actual_host, actual_port


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

    serve_streamable_http(server, args.host, args.port)
