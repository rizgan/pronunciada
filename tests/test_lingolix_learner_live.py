"""bench.third_party.lingolix_learner_live: minute cap, cache, refusals, sentence score, secrets.

No network: lingolix_live's fake transport behind the adapter, synthetic recordings only."""
import json
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("soundfile")

from bench.third_party import live_core as C  # noqa: E402
from bench.third_party import lingolix_learner_live as LX  # noqa: E402

SECRET = "lx_TEST_SECRET_learner_77c1"
FIXTURE = Path(__file__).parent / "data" / "lingolix_v3_check_response.json"


def _wav(p: Path, seconds=0.5, tone=1) -> Path:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 300) for i in range(int(16000 * seconds))))
    return p


def _items(tmp_path, n=3, seconds=30.0):
    return [C.Item("heroico", f"heroico-usma-x{i}-s1", f"x{i}", _wav(tmp_path / f"h{i}.wav", seconds, i + 1),
                   "me gustaría un vaso de agua", "es",
                   meta={"native": i % 2 == 0, "subcorpus": "usma", "prompt_id": "s1"}) for i in range(n)]


def _run(items, root, transport, cap=100.0, key=SECRET):
    ledger = C.Ledger(root / C.LEDGER)
    p = C.plan(LX, items, root, used=0, cap=None)
    budget = C.Budget(cap, 0, ledger, LX.UNIT, LX.NAME)
    st = C.run(LX, p["todo"], transport=transport, key=key, root=root, budget=budget,
               policy=C.Policy(rps=0, base_s=0), sleep=lambda s: None, progress=False)
    return st, ledger


def test_minute_cap_refuses_before_sending(tmp_path):
    t = LX.fake_transport()
    st, ledger = _run(_items(tmp_path, 3, 30.0), tmp_path / "root", t, cap=1.2)       # 0.5 min each
    assert len(t.calls) == 2 and "cap reached" in st["stopped"]
    assert ledger.spent() == pytest.approx(1.0)


def test_request_is_the_lingolix_harness_request(tmp_path):
    t = LX.fake_transport()
    _run(_items(tmp_path, 1, 1.0), tmp_path / "root", t)
    c = t.calls[0]
    assert c["url"] == "https://api.lingolix.com/api/pronunciation/v3/check"
    assert c["data"] == {"sentence": "me gustaría un vaso de agua", "language_code": "es"}
    assert c["auth"] == f"Bearer {SECRET}"


def test_cache_hit_means_no_request(tmp_path):
    items, root = _items(tmp_path, 2, 1.0), tmp_path / "root"
    _run(items, root, LX.fake_transport())
    t = LX.fake_transport()
    st, _ = _run(items, root, t)
    assert t.calls == [] and st["sent"] == 0


@pytest.mark.parametrize("status, body, kind", [
    (401, b"{}", "fatal"), (402, b"{}", "fatal"), (403, b"{}", "fatal"),
    (429, json.dumps({"error": {"message": {"error": "quota_exceeded", "remaining_minutes": 0}}}).encode(), "fatal"),
    (429, b'{"message": "slow down"}', "retry"), (503, b"", "retry"), (400, b'{"message": "bad"}', "fail"),
])
def test_status_classification_is_the_harness_one(status, body, kind):
    assert LX.classify(C.Response(status, {}, body))[0] == kind


def test_quota_error_stops_the_run(tmp_path):
    q = C.Response(429, {}, json.dumps({"error": {"message": {"error": "quota_exceeded"}}}).encode())
    t = LX._FakeAdapter(__import__("bench.third_party.lingolix_live", fromlist=["x"]).FakeTransport(script={1: q}))
    st, _ = _run(_items(tmp_path, 3, 1.0), tmp_path / "root", t)
    assert len(t.calls) == 1 and st["stopped"]


def test_only_heroico_is_allowed():
    for bad in ("speechocean762", "ompal", "l2arctic", "epadb", "saa", "leap", "lingolix_test2"):
        with pytest.raises(C.DatasetRefused):
            C.check_datasets([bad], LX.ALLOWED)
    assert LX.main(["--datasets", "ompal", "--plan"]) == 2


def test_nothing_sent_without_yes_and_cap(tmp_path, monkeypatch):
    items = _items(tmp_path, 1, 1.0)
    monkeypatch.setattr(C, "select_items", lambda *a, **k: items)
    monkeypatch.setattr(C, "live_root", lambda v, dry, base=None: tmp_path / "live")
    monkeypatch.setattr(LX, "real_transport", lambda: pytest.fail("a transport was built"))
    monkeypatch.setenv(LX.KEY_ENV, SECRET)
    assert LX.main(["--minutes-cap", "5"]) == 2
    assert LX.main(["--yes"]) == 2


def test_sentence_score_rule():
    raw = json.loads(FIXTURE.read_text())
    it = C.Item("heroico", "x", "s", Path("x"), raw["text"], "en")
    v = LX.view({"body": json.dumps(raw), "received_utc": "t"}, it)
    assert v["sentence"] == {"total": 71.0, "source": "overall"}
    raw.pop("overall")
    v = LX.view({"body": json.dumps(raw)}, it)
    assert v["sentence"]["source"] == "word mean"
    assert v["sentence"]["total"] == pytest.approx(sum(w["score"] for w in v["words"]) / len(v["words"]))


def test_per_item_file_and_no_credential_anywhere(tmp_path, capsys):
    items, root = _items(tmp_path, 2, 1.0), tmp_path / "root"
    _run(items, root, LX.fake_transport())
    A = C.collect_answers(LX, items, root)
    LX.after_report(A, root)
    rows = [json.loads(x) for x in LX.per_item_path(root).read_text().splitlines()]
    assert [r["id"] for r in rows] == sorted(i.id for i in items)
    assert set(rows[0]) == {"id", "speaker", "sentence_score", "sentence_score_source", "words", "model_version",
                            "received_utc"}
    assert rows[0]["sentence_score_source"] == "overall" and rows[0]["model_version"] == "FAKE-dry-run"
    for p in root.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(encoding="utf-8")
    assert SECRET not in capsys.readouterr().err


def test_does_not_touch_the_lingolix_live_root():
    assert LX.STEM == "lingolix_learner_live" and C.live_root(LX, False).name == "lingolix_learner_live"
