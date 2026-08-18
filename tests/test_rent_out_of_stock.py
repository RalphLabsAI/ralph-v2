"""`available=true` is a catalogue, not a reservation — a create can still refuse for want of stock.

Live round 2, 2026-08-18: hyperstack/montreal-canada-2 advertised an available H100 and answered
the create with `409 OUT_OF_STOCK`. `rent` raised, and the caller's retry loop never saw it — that
loop wraps `wait_ready`, so it covers a box that fails to BOOT and not one that fails to be
CREATED. The round died having rented nothing, with three usable regions left untried.
"""
import pytest

from eval.orchestrator import GpuSpec, ShadeformProvider, _is_out_of_stock

OOS = RuntimeError('shadeform POST /instances/create -> 409: {"error_code":"OUT_OF_STOCK",'
                   '"error":"The instance type [H100] is not available in the region [x]"}')


def test_recognises_the_real_error():
    assert _is_out_of_stock(OOS)
    assert _is_out_of_stock(RuntimeError("... -> 409: whatever they reword it to"))


def test_does_not_swallow_our_own_mistakes():
    """A 401 or a bad body fails the same way everywhere; walking regions would just multiply it."""
    for e in (RuntimeError("-> 401: unauthorized"),
              RuntimeError("-> 400: malformed ssh_key_id"),
              RuntimeError("connection reset")):
        assert not _is_out_of_stock(e)


def _provider(monkeypatch, calls, fail_regions):
    p = ShadeformProvider.__new__(ShadeformProvider)
    p.ssh_key_id = ""
    types = [{"cloud": c, "shade_instance_type": "H100", "hourly_price": price,
              "availability": [{"region": r, "available": True}]}
             for c, r, price in (("a", "r1", 250), ("b", "r2", 330), ("c", "r3", 440))]

    def fake_api(method, path, body=None):
        if method == "GET" and "instances/types" in path:
            return {"instance_types": types}
        if method == "POST":
            calls.append((body["cloud"], body["region"]))
            if body["region"] in fail_regions:
                raise OOS
            return {"id": "inst-" + body["region"]}
        return {"auto_delete": {"date_threshold": "2026-01-01T00:00:00Z"}}

    monkeypatch.setattr(p, "_api", fake_api)
    return p


def test_walks_past_a_region_that_is_out_of_stock(monkeypatch):
    calls = []
    p = _provider(monkeypatch, calls, fail_regions={"r1", "r2"})
    inst = p.rent(GpuSpec(gpu_type="H100"), "ralph-round-9-1")
    assert [r for _c, r in calls] == ["r1", "r2", "r3"]
    assert inst.region == "r3" and inst.id == "inst-r3"


def test_takes_the_first_that_works_and_stops(monkeypatch):
    calls = []
    p = _provider(monkeypatch, calls, fail_regions=set())
    inst = p.rent(GpuSpec(gpu_type="H100"), "ralph-round-9-1")
    assert len(calls) == 1 and inst.region == "r1"


def test_a_non_stock_error_still_raises_immediately(monkeypatch):
    calls = []
    p = ShadeformProvider.__new__(ShadeformProvider)
    p.ssh_key_id = ""
    types = [{"cloud": "a", "shade_instance_type": "H100", "hourly_price": 250,
              "availability": [{"region": "r1", "available": True}]}]

    def fake_api(method, path, body=None):
        if method == "GET" and "instances/types" in path:
            return {"instance_types": types}
        calls.append(1)
        raise RuntimeError("-> 401: unauthorized")

    monkeypatch.setattr(p, "_api", fake_api)
    with pytest.raises(RuntimeError, match="401"):
        p.rent(GpuSpec(gpu_type="H100"), "ralph-round-9-1")
    assert len(calls) == 1, "a bad credential must not be retried across regions"


def test_all_out_of_stock_names_what_it_tried(monkeypatch):
    calls = []
    p = _provider(monkeypatch, calls, fail_regions={"r1", "r2", "r3"})
    with pytest.raises(RuntimeError, match="refused to create for want of stock"):
        p.rent(GpuSpec(gpu_type="H100"), "ralph-round-9-1")
    assert len(calls) == 3
