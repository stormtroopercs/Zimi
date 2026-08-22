#!/usr/bin/env python3
"""MCP streamable-HTTP launcher (zimi.mcp_http) — transport dispatch + wiring.

Standalone unit tests (mirrors tests/test_mcp_chunks.py): they exercise the
launcher's logic without a running server or ZIM files. `zimi.mcp_server` is
imported (it needs the `mcp` extra), so run this with a `mcp`-installed
interpreter, e.g.:

    python3 -m venv /tmp/zm && /tmp/zm/bin/pip install "mcp>=1.0.0"
    /tmp/zm/bin/python tests/test_mcp_http.py

The assertions are version-agnostic (they hold on MCP SDK 1.x and 2.x); where
a behaviour is version-specific, the expected value is derived from the
module's own `_mcp_major()` so the test stays self-consistent on either major.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zimi.mcp_http as mcp_http  # noqa: E402

_SENTINEL_APP = object()


class _FakeServer:
    """Just enough of FastMCP for the launcher's contract tests."""

    def __init__(self):
        self.settings = SimpleNamespace(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=False,
        )
        self.run = MagicMock(name="server.run")
        self.streamable_http_app = MagicMock(
            name="server.streamable_http_app", return_value=_SENTINEL_APP
        )


def _clean_env():
    for var in ("ZIMI_MCP_TRANSPORT", "ZIMI_MCP_HOST", "ZIMI_MCP_PORT", "ZIMI_MCP_PATH"):
        os.environ.pop(var, None)


class TestDispatch(unittest.TestCase):
    """`run_main` picks stdio vs streamable HTTP correctly."""

    def test_stdio_is_default(self):
        _clean_env()
        server = _FakeServer()
        with patch.object(sys, "argv", ["mcp_server"]):
            mcp_http.run_main(server)
        server.run.assert_called_once_with(transport="stdio")
        # The HTTP path must NOT have been entered.
        self.assertFalse(server.streamable_http_app.called)

    def test_http_flag_enters_http_path(self):
        _clean_env()
        server = _FakeServer()
        with patch.object(sys, "argv", ["mcp_server", "--http", "--host", "127.0.0.1",
                                         "--port", "9911", "--path", "/mcp"]), \
             patch.object(mcp_http, "serve_streamable_http") as serve:
            mcp_http.run_main(server)
        serve.assert_called_once_with(server, "127.0.0.1", 9911)
        self.assertFalse(server.run.called, "stdio must not start in --http mode")
        self.assertEqual(os.environ.get("ZIMI_MCP_PATH"), "/mcp")

    def test_http_via_env_var(self):
        _clean_env()
        os.environ["ZIMI_MCP_TRANSPORT"] = "http"
        os.environ["ZIMI_MCP_HOST"] = "0.0.0.0"
        os.environ["ZIMI_MCP_PORT"] = "8100"
        server = _FakeServer()
        with patch.object(sys, "argv", ["mcp_server"]), \
             patch.object(mcp_http, "serve_streamable_http") as serve:
            mcp_host = os.environ.get("ZIMI_MCP_HOST", "0.0.0.0")
            mcp_port = int(os.environ.get("ZIMI_MCP_PORT", "8100"))
            mcp_http.run_main(server)
        serve.assert_called_once_with(server, mcp_host, mcp_port)
        self.assertFalse(server.run.called)

    def test_stdio_explicit_env(self):
        _clean_env()
        os.environ["ZIMI_MCP_TRANSPORT"] = "stdio"
        server = _FakeServer()
        with patch.object(sys, "argv", ["mcp_server"]):
            mcp_http.run_main(server)
        server.run.assert_called_once_with(transport="stdio")

    def test_path_is_normalized_to_leading_slash(self):
        _clean_env()
        server = _FakeServer()
        with patch.object(sys, "argv", ["mcp_server", "--http", "--host", "127.0.0.1",
                                         "--port", "9912", "--path", "mcp"]), \
             patch.object(mcp_http, "serve_streamable_http") as serve:
            mcp_http.run_main(server)
        serve.assert_called_once_with(server, "127.0.0.1", 9912)
        self.assertEqual(os.environ.get("ZIMI_MCP_PATH"), "/mcp")


class TestServeStreamableHttp(unittest.TestCase):
    """`serve_streamable_http` builds the app and drives uvicorn on host:port."""

    def _run(self):
        server = _FakeServer()
        with patch("uvicorn.run") as uv_run:
            mcp_http.serve_streamable_http(server, "127.0.0.1", 9922)
        return server, uv_run

    def test_builds_app_via_streamable_http_app(self):
        server, _ = self._run()
        server.streamable_http_app.assert_called_once()
        self.assertEqual(os.environ.get("ZIMI_MCP_PATH"), "/mcp")

    def test_passes_host_port_to_uvicorn(self):
        _, uv_run = self._run()
        uv_run.assert_called_once()
        args, kwargs = uv_run.call_args
        self.assertIs(args[0], _SENTINEL_APP)
        self.assertEqual(kwargs.get("host"), "127.0.0.1")
        self.assertEqual(kwargs.get("port"), 9922)

    def test_respects_custom_path_from_env(self):
        _clean_env()
        os.environ["ZIMI_MCP_PATH"] = "/custom-mcp"
        server = _FakeServer()
        try:
            with patch("uvicorn.run"):
                mcp_http.serve_streamable_http(server, "127.0.0.1", 9923)
            self.assertEqual(os.environ.get("ZIMI_MCP_PATH"), "/custom-mcp")
        finally:
            _clean_env()


class TestBuildHttpServer(unittest.TestCase):
    """`build_http_server` configures 1.x settings; is a safe no-op on 2.x."""

    def test_noop_and_idempotent_on_current_sdk(self):
        server = _FakeServer()
        before = (
            server.settings.streamable_http_path,
            server.settings.stateless_http,
        )
        # Must not raise on either major, and must be safe to call twice.
        mcp_http.build_http_server(server, "/mcp", True)
        mcp_http.build_http_server(server, "/mcp", True)
        if mcp_http._mcp_major() >= 2:
            # 2.x: values were passed at build time; settings untouched.
            self.assertEqual(
                (server.settings.streamable_http_path, server.settings.stateless_http),
                before,
            )
        else:
            self.assertEqual(server.settings.streamable_http_path, "/mcp")
            self.assertIs(server.settings.stateless_http, True)

    def test_stateless_default_is_on(self):
        self.assertTrue(mcp_http._stateless_default())


class TestMcpMajor(unittest.TestCase):
    def test_returns_positive_int(self):
        self.assertIsInstance(mcp_http._mcp_major(), int)
        self.assertGreaterEqual(mcp_http._mcp_major(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
