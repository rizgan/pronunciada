"""bench.third_party.chivox_live (+ live_core): cap, --yes, cache, ledger, refusals, parsing, secrets.

No test sends anything over the network: every request goes to a local fake transport, and the
datasets are replaced by synthetic recordings."""
import json
import socket
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("soundfile")

from bench.third_party import chivox_live as CX  # noqa: E402
from bench.third_party import live_core as C  # noqa: E402

SECRET = "cvx_TEST_SECRET_9b1f3c55aa"
HAVE_FFMPEG = __import__("shutil").which("ffmpeg") is not None


@pytest.fixture(autouse=True)
def _mp3_without_ffmpeg(monkeypatch):
    """Where ffmpeg is missing, stand in for the encoder with fake MP3 bytes (frame sync + the WAV's
    hash), so the transport tests still run; the encoder itself is tested only where ffmpeg exists."""
    if not HAVE_FFMPEG:
        import hashlib

        monkeypatch.setattr(CX, "encode_audio", lambda it, wav: b"\xff\xfb" + hashlib.sha256(wav).digest())
        monkeypatch.setattr(CX, "record_extra", lambda it, sent: {"sent_audio": {"format": "mp3 (test stand-in)"}})


def _wav(p: Path, seconds=0.5, tone=1) -> Path:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 300) for i in range(int(16000 * seconds))))
    return p


def _items(tmp_path, n=3, lang="en"):
    ds = "speechocean762" if lang == "en" else "ompal"
    text = (lambda i: f"HELLO WORLD NUMBER {i}") if lang == "en" else (lambda i: "他們都很忙"[: 3 + i % 3])
    return [C.Item(ds, f"{ds}-{i:04d}", f"spk{i % 2}", _wav(tmp_path / f"{ds}{i}.wav", tone=i + 1), text(i), lang)
            for i in range(n)]


def _run(items, root, transport, cap=100.0, key=SECRET, used=0.0):
    ledger = C.Ledger(root / C.LEDGER)
    p = C.plan(CX, items, root, used=used, cap=None)
    budget = C.Budget(cap, used, ledger, CX.UNIT, CX.NAME)
    st = C.run(CX, p["todo"], transport=transport, key=key, root=root, budget=budget,
               policy=C.Policy(rps=0, base_s=0), sleep=lambda s: None, progress=False)
    return st, ledger


def _tool_calls(t):
    return [c for c in t.calls if (c["body"] or {}).get("method") == "tools/call"]


# ── cap and consent ──

def test_cap_refuses_before_any_network_call(tmp_path):
    t = CX.FakeTransport()
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t, cap=2)
    assert len(_tool_calls(t)) == 2                         # the third was refused before being sent
    assert st["answered"] == 2 and "cap reached" in st["stopped"]
    assert ledger.spent() == 2


def test_retries_reserve_against_the_cap(tmp_path):
    t = CX.FakeTransport(script={1: C.Response(503, {}, b"busy")})
    st, ledger = _run(_items(tmp_path, 1), tmp_path / "root", t, cap=1)
    assert len(_tool_calls(t)) == 1                         # the retry would cross the cap: not sent
    assert "cap reached" in st["stopped"]
    assert ledger.spent() == 1


def test_cap_counts_points_already_in_the_ledger(tmp_path):
    t = CX.FakeTransport()
    st, _ = _run(_items(tmp_path, 3), tmp_path / "root", t, cap=600, used=599)
    assert len(_tool_calls(t)) == 1


def test_nothing_sent_without_yes_or_cap(tmp_path, monkeypatch, capsys):
    items = _items(tmp_path, 2)
    monkeypatch.setattr(C, "select_items", lambda *a, **k: items)
    monkeypatch.setattr(C, "live_root", lambda v, dry, base=None: tmp_path / ("dry" if dry else "live"))
    monkeypatch.setattr(CX, "real_transport", lambda: pytest.fail("a transport was built"))
    monkeypatch.setenv(CX.KEY_ENV, SECRET)
    assert CX.main(["--datasets", "speechocean762", "--points-cap", "5"]) == 2          # no --yes
    assert CX.main(["--datasets", "speechocean762", "--yes"]) == 2                      # no cap
    monkeypatch.delenv(CX.KEY_ENV)
    assert CX.main(["--datasets", "speechocean762", "--yes", "--points-cap", "5"]) == 2  # no key
    assert CX.main(["--datasets", "speechocean762", "--plan"]) == 0
    err = capsys.readouterr().err
    assert "nothing sent" in err and SECRET not in err
    assert not (tmp_path / "live" / C.LEDGER).exists()


def test_refuses_other_datasets(tmp_path):
    for bad in ("l2arctic", "epadb", "saa", "leap", "common_voice", "heroico", "lingolix_test2"):
        with pytest.raises(C.DatasetRefused):
            C.check_datasets([bad], CX.ALLOWED)
        with pytest.raises(C.DatasetRefused):
            C.load_items(bad, CX.ALLOWED)
    assert CX.main(["--datasets", "epadb", "--plan"]) == 2


# ── cache and ledger ──

def test_cache_hit_means_no_request(tmp_path):
    items, root = _items(tmp_path, 3), tmp_path / "root"
    t1 = CX.FakeTransport()
    _run(items, root, t1)
    assert len(_tool_calls(t1)) == 3
    p = C.plan(CX, items, root, used=0, cap=None)
    assert p["n_send"] == 0 and p["rows"]["speechocean762"]["answered"] == 3
    t2 = CX.FakeTransport()
    st, _ = _run(items, root, t2)
    assert t2.calls == [] and st["sent"] == 0


def test_cached_answer_written_at_once_with_version(tmp_path):
    items, root = _items(tmp_path, 1), tmp_path / "root"
    _run(items, root, CX.FakeTransport())
    files = list((root / "responses").rglob("*.json"))
    assert len(files) == 1
    # server name-version, then the engine's `version` and `res` (as in the real answers)
    assert files[0].name.endswith("__speechocean762-0000__FAKE-chivox-dry-run_FAKE_FAKE.snt.json")
    rec = json.loads(files[0].read_text())
    assert rec["final"] == "answered" and rec["version"] == "FAKE-chivox-dry-run_FAKE_FAKE.snt" and rec["status"] == 200


def test_ledger_arithmetic(tmp_path):
    t = CX.FakeTransport(script={2: C.Response(500, {}, b"oops")})
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t)
    E = ledger.entries()
    att = [e for e in E if e["kind"] == "attempt"]
    res = [e for e in E if e["kind"] == "result"]
    assert len(att) == len(res) == 4                       # 3 items + 1 retry
    assert ledger.spent() == 4 == st["booked"]
    assert [r["status"] for r in res].count(500) == 1
    assert all(a["unit"] == "point" and a["cost"] == 1 and a["id"] and a["utc"] for a in att)
    assert {e["kind"] for e in E} >= {"session", "session_result"}       # initialize logged at 0 cost
    assert sum(e.get("cost") or 0 for e in E if e["kind"] == "session") == 0


def test_unreadable_ledger_fails_closed(tmp_path):
    p = tmp_path / C.LEDGER
    p.write_text('{"kind": "attempt", "cost": 1}\nnot json\n')
    with pytest.raises(C.Fatal, match="cannot be read"):
        C.Ledger(p).spent()


# ── stopping on errors ──

@pytest.mark.parametrize("resp", [
    C.Response(401, {}, b"{}"), C.Response(403, {}, b'{"error": "quota exhausted"}'),
    C.Response(429, {}, b'{"error": "key cap reached"}'),
    C.Response(200, {"Content-Type": "application/json"},
               json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32000,
                                                                "message": "Evaluation failed: insufficient points"}}).encode()),
])
def test_quota_billing_auth_errors_stop_at_once(tmp_path, resp):
    t = CX.FakeTransport(script={1: resp})
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t)
    assert len(_tool_calls(t)) == 1 and st["stopped"]
    assert ledger.spent() == 1


def test_transient_errors_are_retried_and_bounded(tmp_path):
    t = CX.FakeTransport(script={k: C.Response(503, {}, b"") for k in range(1, 20)})
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    assert len(_tool_calls(t)) == C.Policy().max_attempts and st["failed"] == 1


def test_session_expiry_reinitializes(tmp_path):
    t = CX.FakeTransport(script={1: C.Response(404, {}, b"session not found")})
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    assert st["answered"] == 1
    assert sum((c["body"] or {}).get("method") == "initialize" for c in t.calls) == 2


@pytest.mark.parametrize("body", [
    {"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": "Evaluation failed: bad ref_text"}},
    CX.tool_error(2, "评测失败: 请求错误: map[id:12345 msg:something else]"),
])
def test_tool_error_is_not_an_answer(tmp_path, body):
    items, root = _items(tmp_path, 1), tmp_path / "root"
    st, _ = _run(items, root, CX.FakeTransport(script={1: C.Response(200, {}, json.dumps(body).encode())}))
    assert st["failed"] == 1 and st["answered"] == 0
    assert not list((root / "responses").rglob("*.json"))                  # nothing cached as an answer
    assert list((root / "errors").rglob("*.json"))                         # the body is kept for diagnosis
    assert C.plan(CX, items, root, used=0, cap=None)["n_send"] == 1        # it will be sent again next run


# ── the live smoke test of 2026-11-23: MP3 only, "core is timeout" for WAV ──

def test_a_wav_gets_the_timeout_tool_error_from_the_fake():
    t = CX.FakeTransport()
    import base64 as b64
    r = t.post(CX.URL, headers={}, json_body={"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "en_sentence_eval", "arguments": {"ref_text": "HI", "audio_base64": b64.b64encode(b"RIFF....").decode()}}})
    kind, detail = CX.classify(r, {"last_id": 5})
    assert kind == "retry" and CX.TIMEOUT_MSG in detail and CX.attempt_limit(detail) == 2


def test_timeout_is_retried_once_then_given_up(tmp_path):
    to = C.Response(200, {}, json.dumps(CX.tool_error(2, CX.TIMEOUT_TOOL_ERROR), ensure_ascii=False).encode())
    t = CX.FakeTransport(script={1: to, 2: to})
    st, ledger = _run(_items(tmp_path, 2), tmp_path / "root", t)
    assert len(_tool_calls(t)) == 3                       # item 1: 2 attempts; item 2: answered
    assert st["failed"] == 1 and st["answered"] == 1 and st["stopped"] is None
    assert ledger.spent() == 3


def test_first_two_items_timing_out_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(CX, "encode_audio", lambda it, wav: wav)          # what went wrong live: a WAV
    t = CX.FakeTransport()
    st, _ = _run(_items(tmp_path, 4), tmp_path / "root", t)
    assert len(_tool_calls(t)) == 4                       # 2 items x 2 attempts, then the stop
    assert "request format is wrong" in st["stopped"]
    assert not list((tmp_path / "root" / "responses").rglob("*.json"))


def test_mp3_is_sent_and_recorded(tmp_path):
    import base64 as b64
    t = CX.FakeTransport()
    root = tmp_path / "root"
    st, _ = _run(_items(tmp_path, 1), root, t)
    sent = b64.b64decode(_tool_calls(t)[0]["body"]["params"]["arguments"]["audio_base64"])
    assert sent[:4] != b"RIFF" and (sent[:3] == b"ID3" or sent[0] == 0xFF)
    rec = json.loads(next((root / "responses").rglob("*.json")).read_text())
    assert rec["sent_audio"]["format"].startswith("mp3")
    assert rec["request"]["options"]["mp3"]["bitrate"] == "64k"        # the MP3 settings are in the request hash


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_mp3_encoding_is_deterministic_and_recorded(tmp_path):
    it = _items(tmp_path, 1)[0]
    wav = C.audio_bytes(it)
    a = CX.encode_audio(it, wav)
    CX._MP3.clear()
    b = CX.encode_audio(it, wav)
    assert a == b and a[:3] != b"ID3" and a[0] == 0xFF                 # no tags, frame sync first
    ex = CX.record_extra(it, a)["sent_audio"]
    assert ex["ffmpeg"].startswith("ffmpeg version") and ex["ffmpeg_args"] == CX.FFMPEG_ARGS
    assert "64k" in CX.FFMPEG_ARGS and "libmp3lame" in CX.FFMPEG_ARGS


def test_missing_ffmpeg_fails_clearly(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(CX, "_FFV", [])
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(C.Fatal, match="ffmpeg is not installed"):
        CX.ffmpeg_version()


def test_live_shaped_answer_is_read_under_result():
    live = {"app": {"applicationId": "x"}, "audio": {"audioType": "mp3", "channel": 1, "sampleBytes": 2,
                                                     "sampleRate": 16000},
            "ccode": 0, "eof": 1, "request": {"coreType": "en.sent.score", "refText": "HELLO"},
            "result": dict(DOC_EXAMPLE, delaytime=10)}
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en")
    assert CX.view(_rec(live), it)["sentence"]["accuracy"] == 82


# Synthetic answers in the REAL layout of 2026-11-23 (field names and nesting as received; values made up)
def _real(result: dict, core: str) -> dict:
    return {"app": {"applicationId": "x", "userId": "mcp-user"},
            "audio": {"audioType": "mp3", "channel": 1, "sampleBytes": 2, "sampleRate": 16000},
            "ccode": {"code": "c", "len": 1, "time": "0"}, "eof": 1,
            "request": {"coreType": core, "rank": 100, "refText": "x", "tokenId": "t", "attachAudioUrl": 0},
            "result": result, "time": {"request": "0"}, "tokenId": "t", "uuid": "u"}


REAL_EN = _real({"accuracy": 70, "overall": 72, "pron": 74, "integrity": 100, "rank": 100,
                 "fluency": {"overall": 80, "pause": 0, "speed": 100},
                 "rhythm": {"overall": 66, "sense": 100, "stress": 90, "tone": 50},
                 "res": "eng.snt.TEST", "version": "0.0.TEST", "delaytime": 1, "forceout": 0, "info": {"snr": 1.0},
                 "details": [{"char": "sandy", "score": 60, "beginindex": 0, "endindex": 4, "start": 0, "end": 1,
                              "dur": 1, "fluency": 100, "stressscore": 100, "tonescore": 100, "sensescore": 100},
                             {"char": "likes", "score": 90, "beginindex": 6, "endindex": 10},
                             {"char": "skin", "score": 80, "beginindex": 12, "endindex": 15}]}, "en.sent.score")
REAL_ZH = _real({"accuracy": 70, "overall": 72, "pron": 74, "integrity": 100, "tone": 65, "phn": 70,
                 "fluency": {"overall": 80, "pause": 0, "speed": 100}, "chars": {"chn_input": "x"},
                 "res": "chn.snt.TEST", "version": "0.0.TEST",
                 "details": [{"char": "hai", "chn_char": "孩", "score": 61, "tone": 2, "tonescore": 40, "overall": 61,
                              "pron": 61, "phn": 61, "confidence": [0, 1, 90, 5, 4],
                              "phone": [{"char": "h", "score": 50}, {"char": "ai", "score": 70}]},
                             {"char": "er", "chn_char": "儿", "score": 90, "tone": 2, "tonescore": 95,
                              "phone": [{"char": "er", "score": 88}]}]}, "cn.sent.raw")


def test_real_english_layout():
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "SANDY LIKES SKIN", "en")
    v = CX.view(_rec(REAL_EN), it)
    assert v["sentence"] == {"total": 72, "accuracy": 70, "fluency": 80, "prosodic": 66, "completeness": 100,
                             "secondary": 74}
    assert [(w["text"], w["score"], w["phones"]) for w in v["words"]] == [("sandy", 60.0, []), ("likes", 90.0, []),
                                                                          ("skin", 80.0, [])]
    r = C.Response(200, {}, _rec(REAL_EN)["body"].encode())
    assert CX.version(r, {"server": {"name": "chivox-eval", "version": "1.0.0"}}) == "chivox-eval-1.0.0_0.0.TEST_eng.snt.TEST"


def test_real_mandarin_layout_and_alignment_skip_punctuation():
    it = C.Item("ompal", "z", "s", Path("z.wav"), "孩，儿。", "zh")
    v = CX.view(_rec(REAL_ZH), it)
    assert v["zh_units"] == [(0, 1, 61.0, 40.0, 50.0), (2, 3, 90.0, 95.0, 88.0)]    # tonescore; min(initial, final)
    assert v["zh_alignment"] == {"details": 2, "han_in_text": 2, "placed": 2}
    assert v["sentence"]["prosodic"] == 65 and v["sentence"]["secondary"] == 72
    assert [w["text"] for w in v["words"]] == ["孩", "儿"] and v["words"][0]["phones"] == [("h", 50.0), ("ai", 70.0)]


def test_english_without_phones_has_no_sound_row():
    pytest.importorskip("rapidfuzz")          # benchmark-only dependency, absent from the core test environment
    import numpy as np
    from bench.third_party import live_eval as LE

    human = {"so762-1": [{"text": "HI", "phones": ["HH", "AY1"], "accuracy": 8}],
             "so762-2": [{"text": "HI", "phones": ["HH", "AY1"], "accuracy": 4}],
             "so762-3": [{"text": "HI", "phones": ["HH", "AY1"], "accuracy": 6}]}
    views = {k: {"sentence": {}, "words": [{"text": "hi", "score": 10.0 * h[0]["accuracy"], "phones": []}]}
             for k, h in human.items()}
    P = {"y": [], "spk": [], "az": [], "B": [], "key": []}
    for k in human:
        for j in range(2):
            for name, x in (("y", float(j)), ("spk", k), ("az", np.random.rand()), ("B", np.random.rand()),
                            ("key", (k, 0, j))):
                P[name].append(x)
    Wd = {"y": [8.0, 4.0, 6.0], "spk": list(human), "az": [1.0, 2.0, 3.0], "ours": [3.0, 1.0, 2.0],
          "key": [(k, 0) for k in human]}
    S = {k: [] for k in ("accuracy", "fluency", "prosodic", "total", "spk", "key")}
    S.update({f"az_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")},
             ours={k: [] for k in ("accuracy", "fluency", "prosodic", "total")})
    R = LE.eval_so762(views, ref=(P, Wd, S, 0), human=human, n_boot=10, vendor="chivox")
    assert R["sounds_pearson"]["vendor_has_score"] is False and "chivox" not in R["sounds_pearson"]
    assert "chivox" in R["words_pearson"]


def test_fake_answers_in_the_real_layout(tmp_path):
    t = CX.FakeTransport()
    root = tmp_path / "root"
    _run(_items(tmp_path, 1) + _items(tmp_path, 1, "zh"), root, t)
    for f in (root / "responses").rglob("*.json"):
        o = CX.outer(json.loads(f.read_text()))
        assert set(o) == {"app", "audio", "ccode", "eof", "request", "result", "time", "tokenId", "uuid"}
        d0 = o["result"]["details"][0]
        if o["request"]["coreType"] == "cn.sent.raw":
            assert {"chn_char", "tonescore", "phone"} <= set(d0) and set(d0["phone"][0]) == {"char", "score"}
            assert isinstance(o["result"]["tone"], int)
        else:
            assert "phone" not in d0 and set(o["result"]["rhythm"]) == {"overall", "sense", "stress", "tone"}


def test_schema_mismatch_lists_the_keys_received():
    live = {"app": {}, "audio": {}, "ccode": 0, "eof": 1, "request": {},
            "result": {"accuracy": 76, "delaytime": 3, "details": [{"char": "sandy", "score": 70, "beginindex": 0}]}}
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "SANDY", "en")
    with pytest.raises(C.ResponseFormatError) as e:
        CX.view(_rec(live), it)
    m = str(e.value)
    assert "C3" in m and "top level ['app', 'audio', 'ccode', 'eof', 'request', 'result']" in m
    assert "scores object ['accuracy', 'delaytime', 'details']" in m and "beginindex" in m


# ── tools/list ──

def test_schemas_are_checked_and_stored_before_any_audio(tmp_path):
    t = CX.FakeTransport()
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    assert [c["body"].get("method") for c in t.calls[:3]] == ["initialize", "notifications/initialized", "tools/list"]
    assert st["session"]["tool_schemas"]["en_sentence_eval"]["required"] == ["ref_text"]


@pytest.mark.parametrize("change, msg", [
    (lambda s: s["en_sentence_eval"]["required"].append("audio_format"), "requires"),
    (lambda s: s["en_sentence_eval"]["properties"].pop("voiced"), "no longer lists"),
    (lambda s: s.pop("cn_sentence_eval"), "not listed"),
])
def test_schema_change_stops_before_sending_audio(tmp_path, change, msg):
    sch = json.loads(json.dumps(CX.LIVE_SCHEMAS))
    change(sch)
    t = CX.FakeTransport(schemas=sch)
    st, ledger = _run(_items(tmp_path, 2), tmp_path / "root", t)
    assert _tool_calls(t) == [] and msg in st["stopped"] and ledger.spent() == 0


# ── secrets ──

def test_credential_never_written_or_printed(tmp_path, capsys):
    root = tmp_path / "root"
    leak = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -1, "message": f"bad ref for key {SECRET}"}})
    t = CX.FakeTransport(script={1: C.Response(200, {"Authorization": f"Bearer {SECRET}"}, leak.encode())})
    _run(_items(tmp_path, 2), root, t)
    assert all(c["headers"]["Authorization"] == f"Bearer {SECRET}" for c in t.calls)   # it is sent...
    for p in root.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(encoding="utf-8"), p                         # ...never stored
    out = capsys.readouterr()
    assert SECRET not in out.err + out.out


# ── request and response format ──

def test_request_body_follows_the_docs(tmp_path):
    t = CX.FakeTransport()
    _run(_items(tmp_path, 1) + _items(tmp_path, 1, "zh"), tmp_path / "root", t)
    init = t.calls[0]["body"]
    assert init["method"] == "initialize" and t.calls[0]["url"] == "https://mcp-global.cloud.chivox.com/"
    en, zh = (c["body"]["params"] for c in _tool_calls(t))
    assert en["name"] == "en_sentence_eval" and en["arguments"]["accent"] == 3 and en["arguments"]["rank"] == 100
    assert en["arguments"]["voiced"] == 1 and en["arguments"]["attachAudioUrl"] == 0
    assert zh["name"] == "cn_sentence_eval" and zh["arguments"]["attachAudioUrl"] == 0
    assert set(en["arguments"]) == {"ref_text", "audio_base64", "accent", "rank", "voiced", "attachAudioUrl"}
    assert set(zh["arguments"]) == {"ref_text", "audio_base64", "rank", "attachAudioUrl"}         # no age_group
    assert set(zh["arguments"]) <= set(CX.LIVE_SCHEMAS["cn_sentence_eval"]["properties"])
    assert all(c["headers"].get("Mcp-Session-Id") == "fake-session-1" for c in _tool_calls(t))


# README / docs "What the engine returns" example (hello, four phones)
DOC_EXAMPLE = {"overall": 85, "accuracy": 82, "pron": 88, "fluency": {"overall": 78, "speed": 65, "pause": 2},
               "integrity": 95, "details": [{"char": "hello", "score": 85, "phone": [
                   {"phoneme": "h", "score": 90, "dp_type": "normal"}, {"phoneme": "ɛ", "score": 82, "dp_type": "normal"},
                   {"phoneme": "l", "score": 88, "dp_type": "normal"}, {"phoneme": "oʊ", "score": 80, "dp_type": "normal"}]}]}
# /products/mandarin-chinese-assessment example shape
DOC_ZH = {"overall": 78, "pron": 74, "fluency": {"overall": 81, "speed": 72},
          "details": [{"char": "你", "score": 70, "tone_ref": "T3", "tone_detected": "T2", "dp_type": "normal"},
                      {"char": "好", "score": 90, "tone_ref": "T3", "tone_detected": "T3"}]}


def _rec(ev, sse=False):
    m = {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": json.dumps(ev, ensure_ascii=False)}]}}
    if sse:
        return {"status": 200, "headers": {"Content-Type": "text/event-stream"},
                "body": f"event: message\ndata: {json.dumps(m, ensure_ascii=False)}\n\n"}
    return {"status": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(m, ensure_ascii=False)}


@pytest.mark.parametrize("sse", [False, True])
def test_documented_example_maps_to_the_fixed_fields(sse):
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en")
    v = CX.view(_rec(DOC_EXAMPLE, sse), it)
    assert v["sentence"] == {"total": 85, "accuracy": 82, "fluency": 78, "prosodic": None, "completeness": 95,
                             "secondary": 88}
    assert v["words"] == [{"text": "hello", "score": 85.0,
                           "phones": [("h", 90.0), ("ɛ", 82.0), ("l", 88.0), ("oʊ", 80.0)]}]
    r = C.Response(200, _rec(DOC_EXAMPLE, sse)["headers"], _rec(DOC_EXAMPLE, sse)["body"].encode())
    assert CX.classify(r, {"last_id": 2}) == ("answered", "ok")


def test_quickstart_nesting_under_result_is_accepted():
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en")
    assert CX.view(_rec({"result": DOC_EXAMPLE}), it)["sentence"]["total"] == 85


def test_mandarin_units_and_tone_channel():
    it = C.Item("ompal", "z", "s", Path("z.wav"), "你好。", "zh")
    v = CX.view(_rec(DOC_ZH), it)
    assert v["zh_units"] == [(0, 1, 70.0, 0.0, None), (1, 2, 90.0, 100.0, None)]
    assert v["sentence"]["secondary"] == 78 and v["sentence"]["accuracy"] is None


@pytest.mark.parametrize("mutate, assumption", [
    (lambda e: e.__setitem__("overall", 850), "C3"),
    (lambda e: e.pop("details"), "C4"),
    (lambda e: e["details"][0].__setitem__("score", "high"), "C4"),
    (lambda e: e["details"][0]["phone"][0].__setitem__("score", -3), "C5"),
    (lambda e: e["details"][0].__setitem__("char", "zzz"), "C4"),
])
def test_unexpected_format_fails_loudly(mutate, assumption):
    ev = json.loads(json.dumps(DOC_EXAMPLE))
    mutate(ev)
    with pytest.raises(C.ResponseFormatError, match=assumption):
        CX.view(_rec(ev), C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en"))


def test_format_error_stops_run_after_caching(tmp_path):
    bad = dict(DOC_EXAMPLE, overall=850)
    body = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": json.dumps(bad)}]}})
    root = tmp_path / "root"
    st, _ = _run(_items(tmp_path, 2), root, CX.FakeTransport(script={1: C.Response(200, {}, body.encode())}))
    assert "C3" in st["stopped"] and len(list((root / "responses").rglob("*.json"))) == 1


# ── pilot sample and dry-run guard ──

def test_pilot_order_is_seeded_speaker_balanced_and_nested(tmp_path):
    its = [C.Item("speechocean762", f"u{s}{k}", f"s{s}", Path("a"), "x", "en") for s in range(5) for k in range(4)]
    o = C.pilot_order(its, 0)
    assert len({i.speaker for i in o[:5]}) == 5                       # one clip per speaker first
    assert [i.id for i in C.pilot_order(its, 0)] == [i.id for i in o]  # deterministic
    assert [i.id for i in C.pilot_order(its, 1)] != [i.id for i in o]
    assert sorted(i.id for i in o) == sorted(i.id for i in its)


def test_default_pilot_fits_the_free_points():
    assert sum(CX.PILOT.values()) == 600 and CX.cost(3.0) == 1


def test_dry_run_guard_blocks_the_network():
    with C.no_network_guard():
        with pytest.raises(Exception, match="refused by the guard"):
            socket.create_connection(("mcp-global.cloud.chivox.com", 443), timeout=1)


def test_real_transport_only_talks_to_the_documented_host():
    pytest.importorskip("requests")          # benchmark-only dependency, absent from the core test environment
    with pytest.raises(C.Fatal, match="refusing"):
        C.RequestsTransport((CX.HOST,)).post("https://example.org/", headers={}, json_body={})


# ── evaluation on synthetic speechocean762 rows ──

def test_so762_rows_line_up_with_the_scoreboard_rows():
    pytest.importorskip("rapidfuzz")          # benchmark-only dependency, absent from the core test environment
    import numpy as np
    from bench.third_party import live_eval as LE

    rng = np.random.default_rng(0)
    human = {}
    P = {k: [] for k in ("y", "spk", "az", "B", "key")}
    Wd = {k: [] for k in ("y", "spk", "az", "ours", "key")}
    S = {k: [] for k in ("accuracy", "fluency", "prosodic", "total", "spk", "key")}
    S.update({f"az_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")},
             ours={k: [] for k in ("accuracy", "fluency", "prosodic", "total")})
    views = {}
    for u in range(12):
        uid, spk = f"so762-{u:04d}", f"s{u % 4}"
        acc = [float(rng.integers(3, 11)) for _ in range(2)]
        human[uid] = [{"text": "HELLO", "phones": ["HH", "AH0", "L", "OW1"], "accuracy": acc[0]},
                      {"text": "WORLD", "phones": ["W", "ER1", "L", "D"], "accuracy": acc[1]}]
        views[uid] = {"sentence": {"accuracy": 10 * sum(acc), "total": 9 * sum(acc), "fluency": None, "prosodic": None},
                      "words": [{"text": "hello", "score": 10 * acc[0],
                                 "phones": [("h", 1.0), ("ə", 2.0), ("l", 3.0), ("oʊ", 4.0)]},
                                {"text": "world", "score": 10 * acc[1], "phones": []}]}
        for j, a in enumerate(acc):
            for k, v in (("y", a), ("spk", spk), ("az", a + rng.normal()), ("key", (uid, j)), ("ours", a + rng.normal())):
                Wd[k].append(v)
            for k in range(4):
                for name, v in (("y", a / 5), ("spk", spk), ("az", rng.normal()), ("B", rng.normal()), ("key", (uid, j, k))):
                    P[name].append(v)
        for k in ("accuracy", "fluency", "prosodic", "total"):
            S[k].append(sum(acc))
            S[f"az_{k}"].append(rng.normal())
            S["ours"][k].append(rng.normal())
        S["spk"].append(spk)
        S["key"].append(uid)
    answered = dict(list(views.items())[:10])
    R = LE.eval_so762(answered, ref=(P, Wd, S, 0), human=human, n_boot=30, vendor="chivox")
    assert R["coverage"] == {"words": 20, "words_unmatched": 0, "phones": 80, "phones_word_fallback": 40}
    assert R["words_pearson"]["n"] == 20 and R["words_pearson"]["chivox"]["value"] == pytest.approx(1.0)
    assert R["sounds_pearson"]["n"] == 80
    assert R["sentence_accuracy_pearson"]["n"] == 10
    assert R["sentence_accuracy_pearson"]["chivox"]["value"] == pytest.approx(1.0)
    assert not R["sentence_fluency_pearson"]["vendor_has_score"] and "chivox" not in R["sentence_fluency_pearson"]
    assert R["low_word_auc"]["chivox"]["value"] == pytest.approx(1.0)
    assert set(R["words_pearson"]["delta"]) == {"value", "ci", "p", "of"}
