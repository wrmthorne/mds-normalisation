import threading
from http.server import ThreadingHTTPServer

import pytest

from mds_norm.pipeline import extraction as ex
from mds_norm.utils import energy as en


class FakeAgent:
    """The agent's interface without codecarbon or hardware behind it"""

    def __init__(self):
        self.open = None

    def health(self):
        return {
            "ok": True,
            "hardware": {"GPU": {"devices": 1}, "CPU": {"mode": "intel_rapl"}},
            "country": "GBR",
            "session": {"energy_wh": 500.0},
        }

    def start(self, project):
        if self.open:
            raise ValueError(f"a window is already open for {self.open!r}")
        self.open = project
        return {"project": project}

    def stop(self):
        if not self.open:
            raise ValueError("no window is open")
        self.open = None
        return {
            "energy_wh": 900.0,
            "gpu_energy_wh": 700.0,
            "co2_g": 180.0,
            "duration_s": 600.0,
            "session": {"energy_wh": 500.0},
        }


@pytest.fixture
def meter():
    agent = FakeAgent()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), en._handler(agent))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield en.RemoteMeter(f"http://127.0.0.1:{httpd.server_address[1]}", timeout=10.0)
    httpd.shutdown()


def test_meter_round_trips_a_window(meter):
    assert "GPU" in meter.health()["hardware"]
    meter.start("record_fixes_llm")
    assert meter.stop()["energy_wh"] == 900.0


def test_a_second_window_is_an_error_not_a_silent_reopen(meter):
    meter.start("one")
    with pytest.raises(RuntimeError, match="already open"):
        meter.start("two")


def test_stopping_nothing_raises(meter):
    with pytest.raises(RuntimeError, match="no window is open"):
        meter.stop()


def test_an_unreachable_agent_raises_rather_than_reporting_zero():
    with pytest.raises(RuntimeError, match="unreachable"):
        en.RemoteMeter("http://127.0.0.1:1", timeout=1.0).health()


def test_measure_sums_both_hosts(meter, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "EMISSIONS_LOG", tmp_path / "emissions_logs")
    with ex.measure("test", meter.url) as cost:
        pass
    assert cost["energy_wh_server"] == 900.0
    assert cost["energy_wh"] == round(cost["energy_wh_client"] + 900.0, 4)
    assert cost["co2_g"] == round(cost["co2_g_client"] + 180.0, 4)
    assert cost["session_wh_server"] == 500.0


def test_local_runs_keep_the_single_host_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "EMISSIONS_LOG", tmp_path / "emissions_logs")
    with ex.measure("test") as cost:
        pass
    assert "energy_wh_server" not in cost
    assert cost["energy_wh"] >= 0


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:30000/v1", True),
        ("http://127.0.0.1:30000/v1", True),
        ("http://gpu-box:30000/v1", False),
        ("http://192.168.1.20:30000/v1", False),
    ],
)
def test_remote_endpoints_are_recognised(url, local):
    assert ex._is_local(url) is local


def test_a_remote_endpoint_without_an_agent_is_refused():
    args = _args(base_url="http://gpu-box:30000/v1")
    with pytest.raises(SystemExit, match="measures this one"):
        ex._energy_ready(args)


def test_the_refusal_can_be_waived_explicitly():
    ex._energy_ready(_args(base_url="http://gpu-box:30000/v1", allow_unmetered=True))


def test_an_agent_without_a_gpu_is_refused(meter, monkeypatch):
    monkeypatch.setattr(en.RemoteMeter, "health", lambda self: {"hardware": {"CPU": {}}, "country": "GBR"})
    with pytest.raises(SystemExit, match="no GPU"):
        ex._energy_ready(_args(base_url="http://gpu-box:30000/v1", energy_url=meter.url))


def test_summarise_flags_shards_that_measured_only_the_client(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PROGRESS", tmp_path / "progress.jsonl")
    for entry in (
        {"shard": 0, "base_url": "http://gpu-box:30000/v1", "energy_url": None, "energy_wh": 12.0},
        {
            "shard": 1,
            "base_url": "http://gpu-box:30000/v1",
            "energy_url": "http://gpu-box:8770",
            "energy_wh": 950.0,
            "energy_wh_client": 50.0,
            "energy_wh_server": 900.0,
        },
    ):
        ex.journal(entry)
    total = ex._summarise()
    assert total["unmetered_shards"] == 1
    assert total["energy_wh"] == 962.0
    assert total["energy_wh_server"] == 900.0


def _args(**kw):
    import argparse

    return argparse.Namespace(**{"base_url": ex.BASE_URL, "energy_url": None, "allow_unmetered": False, **kw})
