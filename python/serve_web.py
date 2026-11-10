"""Serve web/ locally so the page can fetch its layers and season files.

Browsers block fetch() from file://, so open the map through this server.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/serve_web.py
Then: http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import socketserver

ROOT = pathlib.Path(__file__).resolve().parents[1] / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    with socketserver.TCPServer((args.host, args.port), handler) as srv:
        print(f"serving {ROOT}")
        print(f"open http://{args.host}:{args.port}/")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
