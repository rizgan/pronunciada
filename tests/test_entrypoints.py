"""Developer entry points: `pronunciada check` report, Azure-format output, HTTP server, MCP server.

The report snapshots are rendered from stored results (tests/data/cli/*.json) of the suite's
fox clip (macOS `say`, test_scorer.TEXT) scored against TEXT and against WRONG, so they do not
depend on the TTS voice of the machine. Regenerate the .txt snapshots after an intended
renderer change with PRONUNCIADA_UPDATE_SNAPSHOTS=1.
"""
import asyncio
import base64
import http.client
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error

import pronunciada
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pytest

from pronunciada import report
from pronunciada.compat import azure as AZ
from pronunciada.types import SentenceResult

DATA = Path(__file__).parent / "data" / "cli"
TEXT = "The quick brown fox jumps over the lazy dog"
WRONG = "The quick brown cat jumps over the lazy dog"
UPDATE = bool(os.environ.get("PRONUNCIADA_UPDATE_SNAPSHOTS"))


def _fixture(name):
    d = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    return SentenceResult.from_dict(d), d["coaching"], d


def _snap(name: str, text: str):
    p = DATA / name
    if UPDATE or not p.exists():
        p.write_text(text, encoding="utf-8")
    assert text == p.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- report

@pytest.mark.parametrize("name", ["fox", "fox_wrong"])
@pytest.mark.parametrize("color", [False, True])
def test_report_snapshot(name, color):
    res, cb, _ = _fixture(name)
    out = report.render(res, cb, color=color, width=80)
    _snap(f"{name}.{'color' if color else 'plain'}.txt", out)
    assert ("\x1b[" in out) == color


def test_report_content():
    res, cb, _ = _fixture("fox_wrong")
    out = report.render(res, cb, color=False, width=80)
    lines = out.splitlines()
    assert lines[0].startswith("✓ Good") and "accuracy 8.2/10" in lines[0] and "completeness 89%" in lines[0]
    sent = next(i for i, x in enumerate(lines) if x.strip().startswith("The quick"))
    marks = lines[sent + 1]
    col = lines[sent].index("cat")
    assert marks[col:col + 3] == "___"                      # skipped word underlined
    assert marks[lines[sent].index("The"):][:3] == "~~~"    # middling word
    assert any("not heard" in x for x in lines)
    assert sum(x.strip().startswith("→") for x in lines) == 2   # round 19: no tip on the accepted /ə/ of "The"


def test_report_cjk_alignment_and_extra_speech():
    d = {"lang": "zh", "text": "我们去北京", "prob": 0.3, "strictness": 0.5, "words": [
        {"text": "我们", "prob": 0.95, "start": 0.0, "end": 0.4, "phones": []},
        {"text": "去", "prob": 0.3, "start": 0.4, "end": 0.6,
         "phones": [{"ipa": "tɕʰ", "prob": 0.2, "start": 0.4, "end": 0.5, "heard": "tɕ"},
                    {"ipa": "y", "prob": 0.9, "start": 0.5, "end": 0.6}],
         "syllables": [{"text": "去", "start": 0.4, "end": 0.6, "tone_expected": 4, "tone_heard": 2,
                        "tone_prob": 0.1}]},
        {"text": "北京", "prob": 0.6, "start": 0.6, "end": 1.0, "phones": []}]}
    res = SentenceResult.from_dict(d)
    cb = {"items": [{"kind": "extra_speech", "word_index": None, "after_word": 0, "heard": "a"}], "tips": []}
    out = report.render(res, cb, color=False, width=60)
    lines = out.splitlines()
    assert lines[0].startswith("✗ Try again")
    sent = next(i for i, x in enumerate(lines) if "我们" in x)
    assert lines[sent] == "  我们 + 去 北京"
    assert lines[sent + 1] == "       + ^^ ~~~~"                # wide characters: 2 columns each
    assert any("[tɕʰ→tɕ 20%]" in x for x in lines)
    assert any("tone 去: 4→2 10%" in x for x in lines)


def test_report_never_crashes_on_sparse_results():
    for d in ({"lang": "en", "text": "", "prob": 0.0, "strictness": 0.5},
              {"lang": "en", "text": "hi", "prob": 0.9, "strictness": 0.5, "fluency": None, "completeness": None,
               "scores": {}, "audio": {"issues": ["too_noisy"], "retry_recommended": True},
               "warnings": ["audio is very short (0.10 s)"],
               "words": [{"text": "hi", "prob": 0.1, "start": 0, "end": 0, "omitted": True,
                          "phones": [{"ipa": "h", "prob": 0.1, "start": 0, "end": 0}]}]}):
        out = report.render(SentenceResult.from_dict(d), None, color=True, width=10)
        assert out.endswith("\n")
    assert "too_noisy" not in out and "background noise" in out and "very short" in out


def test_want_color(monkeypatch):
    class TTY(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert report.want_color(TTY()) and not report.want_color(io.StringIO())
    assert not report.want_color(TTY(), no_color=True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert not report.want_color(TTY())
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert report.want_color(io.StringIO())


def test_cell_width():
    assert report.cell_width("abc") == 3 and report.cell_width("北京") == 4
    assert report.cell_width("ə̃") == 1 and report.cell_width("ｱ") == 1 and report.cell_width("ア") == 2


def test_result_dict_roundtrip():
    res, _, d = _fixture("fox")
    again = SentenceResult.from_dict(res.to_dict())
    assert again.to_dict() == res.to_dict()
    assert json.loads(res.to_json())["words"][3]["text"] == "fox"
    p = report.payload(res, coach=True, coaching={"items": [{"gate": 1}], "tips": []}, debug=False)
    assert "features" not in p["words"][0]["phones"][0] and "gate" not in p["coaching"]["items"][0]


# ----------------------------------------------------------------------------- CLI (no model)

def test_cli_langs_version_and_errors(capsys, tmp_path):
    from pronunciada.__main__ import main

    assert main(["langs"]) == 0
    out = capsys.readouterr().out
    assert "zh" in out and "pronunciada[zh]" in out and "ko" in out
    assert main(["langs", "--json"]) == 0
    L = json.loads(capsys.readouterr().out)
    assert {x["code"] for x in L} == {"en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko"}
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and f"pronunciada {pronunciada.__version__}" in capsys.readouterr().out
    assert main(["check", str(tmp_path / "missing.wav"), "hi"]) == 2
    assert "not found" in capsys.readouterr().err


def test_core_import_needs_no_optional_packages():
    code = ("import sys, pronunciada, pronunciada.report, pronunciada.server, pronunciada.compat.azure, "
            "pronunciada.mcp_server, pronunciada.languages, pronunciada.__main__\n"
            "bad = [m for m in ('mcp', 'torch', 'pandas', 'sklearn', 'onnxruntime') if m in sys.modules]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mcp_missing_prints_hint(monkeypatch, capsys):
    from pronunciada import mcp_server

    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    assert mcp_server.main() == 1
    assert "pip install" in capsys.readouterr().err


# ----------------------------------------------------------------------------- Azure format

def test_to_azure_shape_sdk_and_rest():
    res, cb, _ = _fixture("fox_wrong")
    doc = AZ.to_azure(res, coaching_items=cb["items"])
    assert doc["RecognitionStatus"] == "Success" and doc["DisplayText"] == WRONG
    nb = doc["NBest"][0]
    pa = nb["PronunciationAssessment"]
    assert set(pa) == {"AccuracyScore", "FluencyScore", "CompletenessScore", "ProsodyScore", "PronScore"}
    assert pa["AccuracyScore"] == round(100 * res.scores["accuracy"], 1)
    assert pa["CompletenessScore"] == round(100 * res.completeness, 1)
    W = nb["Words"]
    assert [w["Word"] for w in W] == WRONG.split()
    et = [w["PronunciationAssessment"]["ErrorType"] for w in W]
    assert et[0] == "Mispronunciation" and et[3] == "Omission" and et.count("None") == 7
    assert "Phonemes" not in W[3]                            # omitted word: no sound detail
    w0 = W[0]
    assert w0["Offset"] == round(res.words[0].start * 1e7) and isinstance(w0["Duration"], int)
    assert [p["Phoneme"] for p in w0["Phonemes"]] == ["ð", "ə"]
    assert w0["Phonemes"][0]["PronunciationAssessment"]["NBestPhonemes"][0]["Phoneme"] == "ð"
    assert w0["Syllables"][0]["Syllable"] == "ðə"
    assert "Lexical" in nb and "cat" not in nb["Lexical"].split()
    json.dumps(doc)

    rest = AZ.to_azure(res, layout="rest", coaching_items=cb["items"])
    nb = rest["NBest"][0]
    assert "PronunciationAssessment" not in nb and nb["PronScore"] == round(100 * res.scores["total"], 1)
    assert nb["Words"][3]["ErrorType"] == "Omission" and "AccuracyScore" in nb["Words"][0]["Phonemes"][0]


def test_to_azure_options():
    res, cb, _ = _fixture("fox_wrong")
    assert "Words" not in AZ.to_azure(res, granularity="FullText")["NBest"][0]
    w = AZ.to_azure(res, granularity="Word")["NBest"][0]["Words"][0]
    assert "Phonemes" not in w and "Syllables" not in w
    five = AZ.to_azure(res, grading_system="FivePoint")["NBest"][0]["PronunciationAssessment"]
    assert 0 <= five["PronScore"] <= 5
    nomis = AZ.to_azure(res, enable_miscue=False)["NBest"][0]["Words"]
    assert {x["PronunciationAssessment"]["ErrorType"] for x in nomis} <= {"None", "Mispronunciation"}
    # without coaching items (a rebuilt result): word prob < 0.5 -> Mispronunciation, omitted -> Omission
    fb = AZ.to_azure(res)["NBest"][0]["Words"]
    assert fb[3]["PronunciationAssessment"]["ErrorType"] == "Omission"
    items = cb["items"] + [{"kind": "extra_speech", "word_index": None, "after_word": 1, "heard": "ʌm",
                            "start": 0.5, "end": 0.7}]
    W = AZ.to_azure(res, coaching_items=items)["NBest"][0]["Words"]
    assert W[2]["Word"] == "ʌm" and W[2]["PronunciationAssessment"]["ErrorType"] == "Insertion"
    for bad in ({"granularity": "Sentence"}, {"phoneme_alphabet": "SAPI"}, {"layout": "xml"}):
        with pytest.raises(ValueError):
            AZ.to_azure(res, **bad)


def test_azure_header_parsing():
    h = base64.b64encode(json.dumps({"ReferenceText": "hi", "Granularity": "word", "EnableMiscue": "True",
                                     "NBestPhonemeCount": 3}).encode()).decode().rstrip("=")
    p = AZ.parse_header(h)
    assert p["ReferenceText"] == "hi"
    o = AZ.options_from_header(p)
    assert o["granularity"] == "Word" and o["enable_miscue"] is True and o["nbest_phoneme_count"] == 3


# ----------------------------------------------------------------------------- HTTP server

class _StubScorer:
    """Returns the stored fox_wrong result: exercises the HTTP layer without the model."""

    def __init__(self, lang):
        self.lang = lang

    def score(self, audio, text, strictness=0.5):
        if not text.strip():
            raise ValueError("text is empty")
        res, _, _ = _fixture("fox_wrong")
        res.strictness = strictness
        return res


def _wav_bytes(seconds=0.5):
    import soundfile as sf

    buf = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture()
def server():
    from pronunciada.server import make_server

    srv = make_server("127.0.0.1", 0, azure_compat=True, max_bytes=200_000, scorer_factory=_StubScorer)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _req(port, path, data=None, headers=None, method=None):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_native(server):
    st, d = _req(server, "/healthz")
    assert st == 200 and d["status"] == "ok" and d["azure_compat"] is True
    st, d = _req(server, "/v1/languages")
    assert st == 200 and len(d["languages"]) == 10
    wav = _wav_bytes()
    st, d = _req(server, "/v1/score?text=hello&lang=en-US&strictness=0.7", wav, {"Content-Type": "audio/wav"})
    assert st == 200 and d["lang"] == "en" and d["strictness"] == 0.7 and "coaching" not in d
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n{WRONG}\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"coach\"\r\n\r\n1\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{b}--\r\n".encode()
    st, d = _req(server, "/v1/score", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    assert st == 200 and d["text"] == WRONG        # stub has no evidence: coaching cannot be added
    assert "coaching" not in d


def test_server_errors(server):
    wav = _wav_bytes()
    ct = {"Content-Type": "audio/wav"}
    assert _req(server, "/v1/score", wav, ct)[0] == 400                          # no text
    assert _req(server, "/v1/score?text=hi&lang=xx", wav, ct)[0] == 400
    assert _req(server, "/v1/score?text=hi&strictness=2", wav, ct)[0] == 400
    st, d = _req(server, "/v1/score?text=hi", b"not audio", ct)
    assert st == 400 and "decode" in d["error"]
    assert _req(server, "/v1/score?text=hi", b"x" * 300_000, ct)[0] == 413
    assert _req(server, "/v1/score")[0] == 405
    assert _req(server, "/healthz", b"", method="POST")[0] == 405
    assert _req(server, "/nowhere")[0] == 404
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    c.putrequest("POST", "/v1/score?text=hi")
    c.endheaders()
    assert c.getresponse().status == 411


def test_server_azure_compat(server):
    wav = _wav_bytes()
    hdr = base64.b64encode(json.dumps({"ReferenceText": WRONG, "GradingSystem": "HundredMark",
                                       "Granularity": "Phoneme", "EnableMiscue": True}).encode()).decode()
    path = "/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed"
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    c.request("POST", path, body=iter([wav[:1000], wav[1000:]]), encode_chunked=True,
              headers={"Pronunciation-Assessment": hdr, "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000"})
    r = c.getresponse()
    d = json.loads(r.read())
    assert r.status == 200 and d["RecognitionStatus"] == "Success"
    nb = d["NBest"][0]
    assert "PronScore" in nb and nb["Words"][3]["ErrorType"] == "Omission"     # REST layout: flat
    assert _req(server, path, wav, {"Content-Type": "audio/wav"})[0] == 400      # header missing
    bad = base64.b64encode(b'{"ReferenceText": ""}').decode()
    assert _req(server, path, wav, {"Pronunciation-Assessment": bad})[0] == 400


def test_server_without_azure_flag():
    from pronunciada.server import make_server

    srv = make_server("127.0.0.1", 0, scorer_factory=_StubScorer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        st, _ = _req(srv.server_address[1], "/speech/recognition/conversation/cognitiveservices/v1", b"x")
        assert st == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ----------------------------------------------------------------------------- MCP

def test_mcp_tools_listed_in_memory():
    pytest.importorskip("mcp")
    from mcp import Client

    from pronunciada.mcp_server import build_server

    async def run():
        async with Client(build_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("list_languages", {})
            bad = await c.call_tool("assess_pronunciation", {"audio_path": "/nonexistent/a.wav",
                                                             "reference_text": "hi"})
            return names, r, bad
    names, r, bad = asyncio.run(run())
    assert names == {"assess_pronunciation", "list_languages"}
    assert len(r.structured_content["languages"]) == 10 and "Mandarin" in r.content[0].text
    assert bad.is_error and "not found" in bad.content[0].text


def test_mcp_stdio_subprocess():
    pytest.importorskip("mcp")
    try:
        from mcp import Client, StdioServerParameters
    except ImportError:
        pytest.skip("mcp client API differs")

    async def run():
        p = StdioServerParameters(command=sys.executable, args=["-m", "pronunciada", "mcp"])
        async with Client(p) as c:
            return await c.call_tool("list_languages", {})
    r = asyncio.run(run())
    assert not r.is_error and "Korean" in r.content[0].text


# ----------------------------------------------------------------------------- with the model

@pytest.fixture(scope="module")
def fox_wav(tts, tmp_path_factory):
    import soundfile as sf

    p = tmp_path_factory.mktemp("cli") / "fox.wav"
    sf.write(p, tts(TEXT), 16000)
    return p


@pytest.mark.model
def test_check_cli_live(fox_wav, capsys):
    from pronunciada.__main__ import main

    assert main(["check", str(fox_wav), WRONG]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out and "cat" in out and "sentence" in out.splitlines()[0]
    assert main(["check", str(fox_wav), TEXT, "--json", "--coach"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["lang"] == "en" and {"items", "tips", "schema_version"} <= set(d["coaching"])
    assert main(["check", str(fox_wav), " "]) == 2


@pytest.mark.model
def test_score_convenience_default_lang(fox_wav):
    import pronunciada

    r = pronunciada.score(str(fox_wav), TEXT)
    assert r.lang == "en" and pronunciada.score(str(fox_wav), TEXT, lang="en-US").prob == r.prob
    doc = AZ.to_azure(r)                     # live result: miscue tags come from coaching
    assert len(doc["NBest"][0]["Words"]) >= 9


@pytest.mark.model
def test_mcp_assess_live(fox_wav):
    pytest.importorskip("mcp")
    from mcp import Client

    from pronunciada.mcp_server import build_server

    async def run():
        async with Client(build_server()) as c:
            return await c.call_tool("assess_pronunciation", {"audio_path": str(fox_wav), "reference_text": WRONG})
    r = asyncio.run(run())
    assert not r.is_error and r.content[0].text.splitlines()[0].split()[1] in ("Good", "Close", "again")
    d = r.structured_content
    assert d["text"] == WRONG and "coaching" in d and "features" not in d["words"][0]["phones"][0]
    assert json.loads(r.content[1].text)["prob"] == d["prob"]


def test_serve_rejects_unavailable_cuda_before_listening(monkeypatch, capsys):
    import onnxruntime as ort
    from pronunciada import server
    from pronunciada.__main__ import main

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(server, "ThreadingHTTPServer", lambda *args: pytest.fail("opened socket"))
    assert main(["serve", "--device", "cuda"]) == 2
    assert "onnxruntime-gpu" in capsys.readouterr().err
