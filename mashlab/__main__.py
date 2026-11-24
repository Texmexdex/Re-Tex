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


PIP_NAMES = {"multipart": "python-multipart", "python_multipart": "python-multipart", "yaml": "pyyaml",
             "sklearn": "scikit-learn", "cv2": "opencv-python", "soundfile": "soundfile", "beat_this":
             "git+https://github.com/CPJKU/beat_this", "muscriptor": "git+https://github.com/muscriptor/muscriptor.git"}


def _cant_start(e):
    """No quiet fallbacks, also at startup: name what's missing, the real error, and the exact fix."""
    import sys
    mod = (getattr(e, "name", None) or "").split(".")[0]
    print("\n  Re-Tex can't start.", flush=True)
    if isinstance(e, ModuleNotFoundError) and mod:
        pip = PIP_NAMES.get(mod, mod.replace("_", "-"))
        print(f"  The Python library '{mod}' isn't installed in Re-Tex's Python:\n    {sys.executable}", flush=True)
        if mod in ("torch", "torchaudio"):
            print("  Fix (the CUDA build - plain 'pip install torch' is CPU-only on Windows):\n"
                  f'    "{sys.executable}" -m pip install torch==2.8.0 torchaudio==2.8.0 '
                  "--index-url https://download.pytorch.org/whl/cu128", flush=True)
        else:
            print(f'  Fix:\n    "{sys.executable}" -m pip install {pip}', flush=True)
        print("  Everything Re-Tex needs to start is listed in requirements.txt:\n"
              f'    "{sys.executable}" -m pip install -r requirements.txt', flush=True)
    else:
        print(f"  A library failed to load: {type(e).__name__}: {e}\n"
              "  Reinstalling that library usually fixes it (a DLL error with torch: the CUDA build, see "
              "https://pytorch.org/get-started/locally/).", flush=True)
    print(f"\n  The error was: {type(e).__name__}: {e}\n", flush=True)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="Re-Tex")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if _replace_stale(a.port):
        print("  (closed an older copy that was still running in the background)", flush=True)
    try:
        import uvicorn
        from mashlab.server import app
    except ImportError as e:                 # a library Re-Tex needs to start is gone: say which, and the fix
        _cant_start(e)
    port = _free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Re-Tex running at {url}\n  (close this window to quit)\n", flush=True)
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
