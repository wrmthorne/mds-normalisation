from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.output_methods.emissions_data import EmissionsData

DEFAULT_PORT = 8770
CLIENT_TIMEOUT = 120.0  # a window stop takes one measurement cycle, not longer


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tracker(
    project: str, output_dir: str, country: str | None, region: str | None
) -> EmissionsTracker | OfflineEmissionsTracker:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

    kw = {
        "project_name": project,
        "output_dir": output_dir,
        "log_level": "error",
        "tracking_mode": "machine",
        "save_to_api": False,
        "allow_multiple_runs": True,
    }
    if country:
        return OfflineEmissionsTracker(country_iso_code=country, region=region, **kw)
    return EmissionsTracker(**kw)


def _energy(data: EmissionsData) -> dict:
    """One tracker's totals, in Wh and grams"""
    return {
        "energy_wh": round(data.energy_consumed * 1e3, 4),
        "gpu_energy_wh": round(data.gpu_energy * 1e3, 4),
        "cpu_energy_wh": round(data.cpu_energy * 1e3, 4),
        "ram_energy_wh": round(data.ram_energy * 1e3, 4),
        "co2_g": round(data.emissions * 1e3, 4),
        "duration_s": round(data.duration, 1),
    }


class Agent:
    """A machine-mode tracker on the serving host, opened and closed remotely"""

    def __init__(self, output_dir: str, country: str | None = None, region: str | None = None) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # codecarbon won't
        self._output_dir, self._country, self._region = output_dir, country, region
        self._lock = threading.Lock()
        self._window = None
        self._project: str | None = None
        self._started = 0.0
        self.session = _tracker("extraction_session", output_dir, country, region)
        if not getattr(self.session, "_hardware", None):
            raise SystemExit(
                "codecarbon found no measurable hardware on this host — it would "
                "report zeros. Check that nvidia-ml-py is installed and "
                "/sys/class/powercap is readable, then start the agent again."
            )
        self.session.start()

    def hardware(self) -> dict:
        """What is actually being measured, and how"""
        return {
            type(h).__name__: {
                "mode": getattr(h, "_mode", None),
                "devices": getattr(getattr(h, "devices", None), "device_count", None),
                "detail": h.description() if hasattr(h, "description") else repr(h),
            }
            for h in self.session._hardware
        }

    def health(self) -> dict:
        import codecarbon

        session = self.session._prepare_emissions_data()
        return {
            "ok": True,
            "codecarbon": codecarbon.__version__,
            "country": session.country_iso_code,
            "carbon_intensity_g_kwh": round(session.emissions / session.energy_consumed * 1e3, 1)
            if session.energy_consumed
            else None,
            "hardware": self.hardware(),
            "session": _energy(session),
            "window_open": self._project,
        }

    def start(self, project: str) -> dict:
        with self._lock:
            if self._window is not None:
                raise ValueError(f"a window is already open for {self._project!r}")
            self._window = _tracker(project, self._output_dir, self._country, self._region)
            self._project, self._started = project, time.time()
            self._window.start()
        _log(f"window open: {project}")
        return {"project": project, "started": self._started}

    def stop(self) -> dict:
        with self._lock:
            if self._window is None:
                raise ValueError("no window is open")
            self._window.stop()
            out = _energy(self._window.final_emissions_data)
            out["project"], self._window, self._project = self._project, None, None
        # a cumulative session row survives a crashed agent
        self.session.flush()
        out["session"] = _energy(self.session._prepare_emissions_data())
        _log(f"window closed: {out['project']} — {out['energy_wh']:.1f} Wh over {out['duration_s']:.0f}s")
        return out

    def close(self) -> None:
        if self._window is not None:
            self._window.stop()
        self.session.stop()


def _handler(agent: Agent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _route(self, path: str) -> dict:
            if path == "/health":
                return agent.health()
            if path == "/start":
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                return agent.start(payload.get("project") or "extraction")
            if path == "/stop":
                return agent.stop()
            raise KeyError(path)

        def do_GET(self) -> None:
            self._serve()

        def do_POST(self) -> None:
            self._serve()

        def _serve(self) -> None:
            try:
                self._reply(200, self._route(self.path.rstrip("/") or "/health"))
            except KeyError as exc:
                self._reply(404, {"error": f"no such endpoint: {exc.args[0]}"})
            except ValueError as exc:
                self._reply(409, {"error": str(exc)})
            except Exception as exc:  # the client must see it
                self._reply(500, {"error": repr(exc)})

        def log_message(self, *args: object) -> None:
            pass  # the agent logs windows, not every request

    return Handler


def serve(host: str, port: int, output_dir: str, country: str | None, region: str | None) -> None:
    agent = Agent(output_dir, country, region)
    measured = ", ".join(f"{k} ({v['mode'] or 'measured'})" for k, v in agent.hardware().items())
    _log(f"measuring {measured}")
    httpd = ThreadingHTTPServer((host, port), _handler(agent))
    _log(f"agent on http://{host}:{port} — emissions to {output_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("stopping")
    finally:
        agent.close()


class RemoteMeter:
    """The client half: opens and closes the serving host's measurement window"""

    def __init__(self, url: str, timeout: float = CLIENT_TIMEOUT) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read() or b"{}").get("error", exc.reason)
            raise RuntimeError(f"energy agent {path}: {detail}") from None
        except OSError as exc:
            raise RuntimeError(f"energy agent at {self.url} unreachable: {exc}") from None

    def health(self) -> dict:
        return self._call("/health")

    def start(self, project: str) -> None:
        self._call("/start", {"project": project})

    def stop(self) -> dict:
        return self._call("/stop")


def check(url: str) -> dict:
    """The pre-flight the run does for you: an agent that cannot see a GPU is not measuring the model"""
    health = RemoteMeter(url).health()
    hw = health["hardware"]
    print(json.dumps(health, indent=2))
    if "GPU" not in hw:
        raise SystemExit(
            f"the agent at {url} reports no GPU — it would attribute none of the "
            "model's draw. Install nvidia-ml-py on the serving host and restart it."
        )
    cpu_mode = (hw.get("CPU") or {}).get("mode")
    if cpu_mode and cpu_mode not in ("intel_rapl", "intel_power_gadget", "psys"):
        print(
            f"note: CPU energy is {cpu_mode!r} — an estimate, not a measurement. "
            "The GPU dominates this workload, so this is a small term."
        )
    return health


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Energy accounting when the model endpoint is served from another machine."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the agent on the serving host")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--output-dir", default="emissions_logs", help="where this host's emissions.csv is written")
    p.add_argument(
        "--country-iso-code", help="3-letter code, e.g. GBR — measures offline, without the geolocation call"
    )
    p.add_argument("--region", help="province/state, offline mode only")

    p = sub.add_parser("check", help="print what an agent can measure")
    p.add_argument("url")

    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.host, args.port, args.output_dir, args.country_iso_code, args.region)
    else:
        check(args.url)


if __name__ == "__main__":
    main()
