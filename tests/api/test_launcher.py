"""``eicvibe-web``: a clear error for a busy port, and a browser that opens only once the server answers."""

import http.server
import socket
import threading

import pytest

pytest.importorskip("fastapi")

from eicvibe.api.__main__ import main, open_when_ready, port_is_free  # noqa: E402


def test_a_busy_port_is_reported_before_starting(tmp_path, capsys):
    from eicvibe.api.demo import write_demo

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not port_is_free("127.0.0.1", port)
        with pytest.raises(SystemExit):
            main([str(write_demo(tmp_path)), "--port", str(port)])
    assert f"port {port} is in use" in capsys.readouterr().err


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200 if self.path == "/api/v1/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_the_browser_opens_once_the_server_answers():
    opened = []
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    waiting = open_when_ready(url, timeout=10.0, opener=opened.append)
    waiting.join(0.5)
    assert opened == []  # nothing listens yet: do not open a page that would fail
    server = http.server.HTTPServer(("127.0.0.1", port), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        waiting.join(15.0)
        assert opened == [url]
    finally:
        server.shutdown()
