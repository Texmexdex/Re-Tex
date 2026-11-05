"""python -m mashlab  ->  starts the local server and opens the UI in your browser."""
import argparse
import socket
import threading
import time
import webbrowser
import warnings

warnings.filterwarnings("ignore")


def _free_port(preferred):
    for p in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0


def _replace_stale(port):
    """If an older Mashup Lab is still holding this port (e.g. left running in the background), ask
    it to quit and wait for the port to free up. Anything else on the port is left alone."""
    import json
    import urllib.request
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/api/whoami", timeout=1.5) as r:
            if json.loads(r.read()).get("app") != "mashup_lab":
                return False
    except Exception:
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/api/shutdown", data=b"{}", method="POST"), timeout=3)
    except Exception:
        pass
    for _ in range(40):                      # up to ~8 s
        time.sleep(0.2)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                continue
    return False


def main():
    ap = argparse.ArgumentParser(description="Re-Tex")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if _replace_stale(a.port):
        print("  (closed an older copy that was still running in the background)", flush=True)
    import uvicorn
    from mashlab.server import app
    port = _free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Re-Tex running at {url}\n  (close this window to quit)\n", flush=True)
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
