'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Desktop Launcher Module.

Runs the CL3O visualizer as a standalone desktop application. It starts (or
reuses) the FastAPI server that serves both the JSON API and the built React
UI on a single port, then opens it in a native OS window (pywebview) or, as a
fallback, a chromeless Chrome/Edge "--app" window. No browser tab and no
separate dev server are required.

Usage (from the project root):
    python -m cl3o.ui.desktop

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import sys
import asyncio
import time
import threading
import subprocess
import urllib.request
import uvicorn
from pathlib import Path

# ================ Pathing ================


# ================ Module imports ================

# FastAPI application (serves the JSON API and the built SPA)
from cl3o.ui.backend.app import app

# ================ Module constants ================
_HOST  = "127.0.0.1"
_PORT  = 8000
_URL   = f"http://{_HOST}:{_PORT}"
_TITLE = "CL3O - Wing Structural Visualizer"

_BROWSER_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


# ========================================================================
# PRIVATE API - Server lifecycle
# ========================================================================

def _server_is_up(timeout: float = 1.0) -> bool:
    '''Return True when the API health endpoint already responds.'''
    try:
        with urllib.request.urlopen(f"{_URL}/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _start_server() -> "uvicorn.Server":
    '''Run uvicorn in a daemon thread; return the Server for graceful shutdown.'''
    config = uvicorn.Config(app, host=_HOST, port=_PORT, log_level="warning")
    server = uvicorn.Server(config)

    def _serve() -> None:
        asyncio.run(server.serve())

    threading.Thread(target=_serve, daemon=True).start()
    return server


def _wait_until_up(timeout: float = 25.0) -> bool:
    '''Poll the health endpoint until it responds or the timeout elapses.'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_is_up():
            return True
        time.sleep(0.3)
    return False


def _shutdown(server: "uvicorn.Server | None") -> None:
    '''Signal uvicorn to stop and exit the process.'''
    if server is not None:
        server.should_exit = True
    sys.exit(0)


# ========================================================================
# PRIVATE API - Window opening
# ========================================================================

def _open_native_window(server: "uvicorn.Server | None") -> bool:
    '''Open a native OS webview window; return False if unavailable.'''
    try:
        import webview
    except Exception:
        return False
    win = webview.create_window(_TITLE, _URL, width=1480, height=900)
    win.events.closed += lambda: _shutdown(server)
    webview.start()
    return True


def _find_browser() -> str | None:
    '''Return the path to an installed Chromium browser, or None.'''
    for path in _BROWSER_PATHS:
        if Path(path).is_file():
            return path
    return None


def _open_browser_app(browser: str) -> subprocess.Popen:
    '''Open a chromeless Chrome/Edge "--app" window at the UI URL.'''
    return subprocess.Popen([
        browser,
        f"--app={_URL}",
        "--window-size=1480,900",
    ])


# ========================================================================
# PUBLIC API - Entry point
# ========================================================================

def main() -> None:
    '''Start (or reuse) the server and open the desktop window.'''
    sys.stdout.write(
        "[CL3O] Tip: after changing frontend source files, rebuild with:\n"
        "|   PowerShell : cd src/cl3o/ui/frontend; npm run build\n"
        "|   bash/cmd   : cd src/cl3o/ui/frontend && npm run build\n"
    )

    server: "uvicorn.Server | None" = None

    if not _server_is_up():
        server = _start_server()
        if not _wait_until_up():
            raise RuntimeError(
                f"[CL3O] Backend did not come up on {_URL}.\n"
                f"| Build the frontend first : npm run build (in src/cl3o/ui/frontend)\n"
                f"| And make sure port {_PORT} is free."
            )

    # Prefer a native window; fall back to a chromeless browser window.
    if _open_native_window(server):
        return

    browser = _find_browser()
    if browser is not None:
        _open_browser_app(browser).wait()          # block until window closes
        _shutdown(server)

    sys.stdout.write(
        f"[CL3O] No native window or Chromium browser found.\n"
        f"| Open {_URL} in any browser. Press Ctrl+C to stop the server.\n"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _shutdown(server)


if __name__ == "__main__":
    main()
