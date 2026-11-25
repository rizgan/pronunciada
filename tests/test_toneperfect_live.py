"""bench.third_party.toneperfect_live: credit rule, cap, test keys, error mapping, parsing, secrets.

No network: a local fake transport and synthetic recordings only."""
import json
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("soundfile")

from bench.third_party import live_core as C  # noqa: E402
from bench.third_party import toneperfect_live as TP  # noqa: E402

SECRET = "tp_live_TEST_SECRET_51ac0e9d"


def _wav(p: Path, seconds=0.5, tone=1) -> Path:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 300) for i in range(int(16000 * seconds))))
    return p


def _items(tmp_path, n=3, seconds=0.5, ds="speechocean762", lang="en"):
    return [C.Item(ds, f"{ds}-{i}", f"spk{i}", _wav(tmp_path / f"{ds}{i}.wav", seconds, i + 1),
                   f"HELLO WORLD NUMBER {i}" if lang != "zh" else "他們都很忙", lang) for i in range(n)]


def _run(items, root, transport, cap=100.0, key=SECRET):
    ledger = C.Ledger(root / C.LEDGER)
    p = C.plan(TP, items, root, used=0, cap=None)
    budget = C.Budget(cap, 0, ledger, TP.UNIT, TP.NAME)
    st = C.run(TP, p["todo"], transport=transport, key=key, root=root, budget=budget,
               policy=C.Policy(rps=0, base_s=0), sleep=lambda s: None, progress=False)
    return st, ledger


def _err(status, code, headers=None, **extra):
    return C.Response(status, headers or {}, json.dumps({"error": {"code": code, "message": code}, **extra}).encode())


@pytest.mark.parametrize("sec, credits", [(0.2, 1), (15.0, 1), (15.01, 2), (30.0, 2), (30.5, 3), (45.0, 3),
                                          (46.0, 4), (119.0, 8), (120.0, 8)])
def test_credit_rule(sec, credits):
    assert TP.cost(sec) == credits


def test_cap_in_credits_refuses_before_sending(tmp_path):
    t = TP.FakeTransport()
    st, ledger = _run(_items(tmp_path, 3, seconds=16.0), tmp_path / "root", t, cap=5)   # 2 credits each
    assert len(t.calls) == 2 and "cap reached" in st["stopped"] and ledger.spent() == 4


def test_default_pilot_fits_fifty_credits():
    assert sum(TP.PILOT.values()) == 48 <= 50


def test_test_keys_are_refused(tmp_path, monkeypatch):
    assert "test key" in TP.check_key("tp_test_abc123")
    assert TP.check_key(SECRET) is None
    items = _items(tmp_path, 1)
    monkeypatch.setattr(C, "select_items", lambda *a, **k: items)
    monkeypatch.setattr(C, "live_root", lambda v, dry, base=None: tmp_path / "live")
    monkeypatch.setattr(TP, "real_transport", lambda: pytest.fail("a transport was built"))
    monkeypatch.setenv(TP.KEY_ENV, "tp_test_abc123")
    assert TP.main(["--datasets", "speechocean762", "--yes", "--credits-cap", "5"]) == 2


def test_test_fixture_answer_stops_the_run(tmp_path):
    fixture = C.Response(200, {"X-TonePerfect-Test": "true"}, json.dumps({"test": True, "response_version": 2}).encode())
    t = TP.FakeTransport(script={1: fixture})
    st, _ = _run(_items(tmp_path, 3), tmp_path / "root", t)
    assert len(t.calls) == 1 and "test-key" in st["stopped"]
    assert not list((tmp_path / "root" / "responses").rglob("*.json"))      # never cached as an answer


@pytest.mark.parametrize("resp, kind", [
    (_err(401, "invalid_api_key"), "fatal"), (_err(402, "insufficient_credits"), "fatal"),
    (_err(402, "subscription_required"), "fatal"), (_err(429, "free_capacity"), "fatal"),
    (_err(429, "rate_limit"), "retry"), (_err(429, "account_busy", {"Retry-After": "1"}), "retry"),
    (_err(503, "engine_unavailable"), "retry"), (_err(504, "result_held"), "retry"),
    (_err(409, "request_pending"), "retry"), (_err(409, "idempotency_conflict"), "fail"),
    (_err(422, "invalid_assessment"), "rejected"), (_err(413, "upload_too_large"), "rejected"),
    (C.Response(200, {}, json.dumps({"response_version": 2, "assessable": False, "reason": "no_speech"}).encode()),
     "rejected"),
    (C.Response(200, {}, json.dumps({"response_version": 1, "assessable": True}).encode()), "retry"),
])
def test_status_mapping(resp, kind):
    assert TP.classify(resp)[0] == kind


def test_billing_error_stops_immediately(tmp_path):
    t = TP.FakeTransport(script={1: _err(402, "insufficient_credits")})
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t)
    assert len(t.calls) == 1 and "billing" in st["stopped"]


def test_request_headers_pin_the_version_and_reuse_one_idempotency_key(tmp_path):
    t = TP.FakeTransport(script={1: _err(503, "engine_unavailable")})
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    a, b = t.calls
    assert a["headers"]["Idempotency-Key"] == b["headers"]["Idempotency-Key"]       # a retry is the same request
    assert a["headers"]["TonePerfect-Version"] == "2" and a["headers"]["TonePerfect-Fallback"] == "off"
    assert a["url"] == "https://api.toneperfect.app/v1/assess"
    assert a["data"] == {"text": "HELLO WORLD NUMBER 0", "language": "en-US"}
    assert st["answered"] == 1


def test_cache_keyed_by_model_and_response_version_and_never_resent(tmp_path):
    items, root = _items(tmp_path, 2), tmp_path / "root"
    _run(items, root, TP.FakeTransport())
    names = sorted(p.name for p in (root / "responses").rglob("*.json"))
    assert all(n.endswith("__FAKE-dry-run+scoremap-5__rv2.json") for n in names) and len(names) == 2
    t = TP.FakeTransport()
    st, _ = _run(items, root, t)
    assert t.calls == [] and st["sent"] == 0


def test_same_request_in_two_sets_is_sent_once(tmp_path):
    a = _items(tmp_path, 2)
    b = [C.Item("lingolix_test2", x.id, x.speaker, x.audio, x.text, "en") for x in a]
    p = C.plan(TP, a + b, tmp_path / "root", used=0, cap=None)
    assert p["n_send"] == 2 and p["rows"]["lingolix_test2"]["same_request_as_another_set"] == 2


def test_credential_never_stored(tmp_path, capsys):
    root = tmp_path / "root"
    t = TP.FakeTransport(script={1: _err(422, f"invalid_reference {SECRET}")})
    _run(_items(tmp_path, 2), root, t)
    for p in root.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(encoding="utf-8")
    assert SECRET not in capsys.readouterr().err


# ── reading answers ──

def test_word_reduction_follows_the_lingolix_adapter():
    raw = {"words": [{"word": "a", "score": 90, "decision": "no_correction"},
                     {"word": "b", "score": None, "decision": "partial"},
                     {"word": "c", "score": 30, "decision": "review"},
                     {"word": "d", "score": 60, "decision": "uncertain"}],
           "raw": {"words": [{"score": 91}, {"score": 77}, {"score": 31}, {"score": 61}]}}
    w = TP.toneperfect_words(raw)
    assert [x["score"] for x in w] == [0.9, 0.77, 0.3, 0.6]
    assert [x["error_type"] for x in w] == [None, None, "Mispronunciation", "Mispronunciation"]


# Built from the documented field names (/docs, "Response structure", version 2 full body)
DOC_EN = {"id": "asm_1", "language": "en", "locale": "en-US", "model_version": "english-v2-abc+scoremap-5",
          "response_version": 2, "task": "general", "status": "scored", "assessable": True, "reason": None,
          "scores": {"overall": 82.5, "pronunciation": 80.0, "integrity": 100, "fluency": None, "rhythm": None},
          "words": [{"word": "HELLO", "score": 95, "decision": "no_correction", "omitted": False,
                     "phones": [{"expected": "h", "heard": "h", "score": 97, "operation": "match"},
                                {"expected": "ə", "heard": "ɛ", "score": 70, "operation": "substitute"}]},
                    {"word": "WORLD", "score": 40, "decision": "review", "omitted": False, "phones": []},
                    {"word": "NUMBER", "score": None, "decision": "partial", "phones": []},
                    {"word": "0", "score": 0, "decision": "review", "omitted": True, "phones": []}],
          "raw": {"words": [{"score": 95}, {"score": 40}, {"score": 66}, {"score": 0}]},
          "billing": {"credits": 1, "balance_credits": 49}}
DOC_ZH = {"model_version": "mandarin-human-v2-85741cdf3967-fbf73d85ee88+scoremap-5", "response_version": 2,
          "assessable": True, "scores": {"overall": 99.6, "pronunciation": 97.8, "initial": 98.2, "final": 97.3,
                                         "tone": 98, "integrity": 100, "fluency": None, "rhythm": None},
          "words": [{"word": "你好", "score": 98, "pinyin": "ni3 hao3", "phones": []}],
          "syllables": [{"index": 0, "score": 97, "components": {"initial": {"score": 99}, "final": {"score": 96},
                                                                 "tone": {"score": 98, "expected": 3, "heard": 3}}},
                        {"index": 1, "score": 95, "components": {"initial": {"score": 93}, "final": {"score": 97},
                                                                 "tone": {"score": 60, "expected": 3, "heard": 2}}}],
          "billing": {"credits": 1, "balance_credits": 49}}


def test_english_view_uses_the_fixed_fields():
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO WORLD NUMBER 0", "en")
    v = TP.view({"body": json.dumps(DOC_EN)}, it)
    assert v["sentence"] == {"total": 82.5, "accuracy": 80.0, "completeness": 100, "fluency": None, "prosodic": None,
                             "secondary": None}
    assert [w["score"] for w in v["words"]] == [95.0, 40.0, 66.0, 0.0]
    assert [w["flag"] for w in v["words"]] == [False, True, False, True]          # review / uncertain only
    assert v["words"][0]["phones"] == [("h", 97.0), ("ə", 70.0)]


def test_mandarin_view_maps_syllables_to_characters():
    it = C.Item("ompal", "z", "s", Path("z.wav"), "你好", "zh")
    v = TP.view({"body": json.dumps(DOC_ZH)}, it)
    assert v["zh_units"] == [(0, 1, 97, 98, 96), (1, 2, 95, 60, 93)]
    assert v["sentence"]["prosodic"] == 98 and v["sentence"]["secondary"] == 99.6


def test_unassessable_is_not_scored():
    it = C.Item("ompal", "z", "s", Path("z.wav"), "你好", "zh")
    assert TP.view({"body": json.dumps({"response_version": 2, "assessable": False})}, it) is None


@pytest.mark.parametrize("mutate, assumption", [
    (lambda b: b["scores"].__setitem__("overall", 0.82), None),          # 0-1 values are still in range
    (lambda b: b["scores"].__setitem__("overall", 820), "T1"),
    (lambda b: b.__setitem__("response_version", 1), "T3"),
    (lambda b: b.pop("words"), "T2"),
    (lambda b: [w.__setitem__("word", "zz") for w in b["words"]], "T2"),
])
def test_format_assumptions(mutate, assumption):
    b = json.loads(json.dumps(DOC_EN))
    mutate(b)
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO WORLD NUMBER 0", "en")
    if assumption is None:
        TP.view({"body": json.dumps(b)}, it)
    else:
        with pytest.raises(C.ResponseFormatError, match=assumption):
            TP.view({"body": json.dumps(b)}, it)


def test_allowed_sets():
    assert set(TP.ALLOWED) == {"speechocean762", "ompal", "heroico", "lingolix_test2",
                               *(f"lingolix_test1_{x}" for x in ("en", "fr", "es", "de", "ja", "ko"))}
    for bad in ("l2arctic", "epadb", "saa", "leap", "common_voice"):
        with pytest.raises(C.DatasetRefused):
            C.check_datasets([bad], TP.ALLOWED)


def test_heroico_pilot_order_keeps_prompt_pairs():
    its = [C.Item("heroico", f"{p}-{s}", s, Path("a"), "x", "es",
                  meta={"prompt_id": p, "native": s.startswith("n"), "subcorpus": "usma" if s[1] != "h" else "heroico"})
           for p in ("s1", "s2", "s3") for s in ("nu1", "nu2", "lu1", "lu2", "lu3", "nh1")]
    o = C.pilot_order(its, 0)
    first = o[:4]
    assert len({i.meta["prompt_id"] for i in first}) == 1
    assert {i.meta["native"] for i in first} == {True, False}
