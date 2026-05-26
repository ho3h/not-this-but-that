#!/usr/bin/env python3
"""Thin client for the probe daemon.

Usage:
    probe.py '{"cmd": "ping"}'
    echo '{"cmd": "graph", "query": "decoder_neighbors", "anchor": 3223, "k": 10}' | probe.py
    probe.py @path/to/body.json   # @-prefix reads body from a file

Use jq to pretty-print: probe.py '...' | jq .
"""

import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8765/probe"
TIMEOUT_S = 1800  # ladder runs can be minutes


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        body = open(arg[1:]).read() if arg.startswith("@") else arg
    else:
        body = sys.stdin.read()
    body = body.strip()
    try:
        json.loads(body)  # validate
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"client: bad json: {e}"}))
        sys.exit(2)
    req = urllib.request.Request(
        URL, data=body.encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            print(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(e.read().decode())
        sys.exit(1)
    except urllib.error.URLError as e:
        print(json.dumps({"ok": False, "error": f"client: connect failed: {e.reason}"}))
        sys.exit(3)


if __name__ == "__main__":
    main()
