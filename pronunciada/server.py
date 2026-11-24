"""Local HTTP scoring server (standard library only). A development server: no authentication,
no TLS; it binds to 127.0.0.1 by default. Do not expose it to a network you do not trust.

    pronunciada serve [--host 127.0.0.1] [--port 8765] [--azure-compat] [--max-mb 20]

Endpoints
  GET  /healthz                 {"status": "ok", "version": ...}
  GET  /v1/languages            pronunciada.languages.languages()
  POST /v1/score                audio as multipart/form-data (file field "audio") or as the raw
                                request body (audio/wav ...); fields text, lang (default en),
                                coach (0/1), strictness (0-1), sensitivity, max_tips as form
                                fields or query parameters -> pronunciada.report.payload JSON
  POST /speech/recognition/{conversation|interactive|dictation}/cognitiveservices/v1
        (only with azure_compat) Azure short-audio REST shape: ?language=en-US&format=detailed,
        header Pronunciation-Assessment = base64 JSON with ReferenceText -> Azure-shaped JSON
        (pronunciada.compat.azure, REST layout; the numbers are pronunciada's own)

Errors are JSON {"error": message} with 400 (bad input), 404, 405, 411, 413 (body larger than
max_bytes), 422 (audio could not be scored), 501 (language extra not installed), 500.
"""
from __future__ import annotations

import json
import sys
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

AZURE_PATHS = tuple(f"/speech/recognition/{m}/cognitiveservices/v1"
                    for m in ("conversation", "interactive", "dictation"))
DEFAULT_MAX_BYTES = 20_000_000


class HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _lang(code: str) -> str:
    from .text import normalize_lang

    try:
        return normalize_lang(code)
    except ValueError:
        try:
            return normalize_lang(code.replace("_", "-").split("-")[0])
        except ValueError as e:
            raise HTTPError(400, str(e)) from None


def _float(v, name: str, default: float) -> float:
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except ValueError:
        raise HTTPError(400, f"{name} must be a number") from None
    if not 0.0 <= x <= 1.0:
        raise HTTPError(400, f"{name} must be between 0 and 1")
    return x


def _flag(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if v is not None else False


class App:
    """The request logic, independent of the HTTP plumbing (and testable on its own)."""

    def __init__(self, azure_compat: bool = False, max_bytes: int = DEFAULT_MAX_BYTES, threads: int | None = None,
                 scorer_factory=None, *, precision: str = "fp32", device: str = "cpu"):
        self.azure_compat = azure_compat
        self.max_bytes = int(max_bytes)
        self.threads = threads
        self.precision = precision
        self.device = device
        self._lock = threading.Lock()         # one scoring at a time: bounded memory / CPU
        self._scorers: dict = {}
        self._factory = scorer_factory

    def scorer(self, lang: str):
        if lang not in self._scorers:
            if self._factory is not None:
                self._scorers[lang] = self._factory(lang)
            else:
                from .scorer import Scorer

                self._scorers[lang] = Scorer(lang, threads=self.threads, precision=self.precision, device=self.device)
        return self._scorers[lang]

    def score(self, audio: bytes, text: str, lang: str, strictness: float):
        from .audio import load_audio

        if not text or not text.strip():
            raise HTTPError(400, "text is required")
        if not audio:
            raise HTTPError(400, "audio is required")
        try:
            wav = load_audio(audio)
        except Exception as e:                # noqa: BLE001  (soundfile raises several types)
            raise HTTPError(400, f"could not decode the audio ({type(e).__name__}); send WAV / FLAC / OGG") from None
        with self._lock:
            try:
                sc = self.scorer(lang)
            except ImportError as e:
                from .languages import EXTRAS

                extra = EXTRAS.get(lang, (None,))[0]
                hint = f"; pip install 'pronunciada[{extra}]'" if extra else ""
                raise HTTPError(501, f"{lang} needs an optional package: {e}{hint}") from None
            try:
                return sc.score((wav, 16000), text, strictness=strictness)
            except ValueError as e:
                raise HTTPError(422, str(e)) from None

    # ------------------------------------------------------------------------- endpoints

    def native(self, body: bytes, ctype: str, query: dict) -> dict:
        from .report import payload

        fields = {k: v[-1] for k, v in query.items()}
        audio = body
        if ctype.lower().startswith("multipart/form-data"):
            form, files = parse_multipart(body, ctype)
            fields.update(form)
            audio = files.get("audio") or (next(iter(files.values())) if files else b"")
        elif ctype.lower().startswith("application/x-www-form-urlencoded"):
            raise HTTPError(400, "send the audio as multipart/form-data or as the raw request body")
        lang = _lang(fields.get("lang") or "en")
        strict = _float(fields.get("strictness"), "strictness", 0.5)
        res = self.score(audio, fields.get("text", ""), lang, strict)
        sens = fields.get("sensitivity") or "balanced"
        if sens not in ("careful", "balanced", "eager"):
            raise HTTPError(400, "sensitivity must be careful, balanced or eager")
        try:
            max_tips = int(fields.get("max_tips") or 3)
        except ValueError:
            raise HTTPError(400, "max_tips must be an integer") from None
        return payload(res, coach=_flag(fields.get("coach")), sensitivity=sens, max_tips=max_tips)

    def azure(self, body: bytes, headers, query: dict) -> dict:
        from .compat import azure as AZ

        h = headers.get("Pronunciation-Assessment")
        if not h:
            raise HTTPError(400, "missing Pronunciation-Assessment header (base64 JSON with ReferenceText)")
        try:
            params = AZ.parse_header(h)
        except (ValueError, UnicodeDecodeError) as e:
            raise HTTPError(400, f"bad Pronunciation-Assessment header: {e}") from None
        text = params.get("ReferenceText") or ""
        if not str(text).strip():
            raise HTTPError(400, "ReferenceText is required (pronunciada scores against a known text)")
        loc = (query.get("language") or ["en-US"])[-1]
        res = self.score(body, str(text), _lang(loc), 0.5)
        try:
            opts = AZ.options_from_header(params)
            return AZ.to_azure(res, layout="rest", **opts)
        except ValueError as e:
            raise HTTPError(400, str(e)) from None


def parse_multipart(body: bytes, ctype: str) -> tuple[dict, dict]:
    """multipart/form-data -> ({field: text}, {field: bytes}) with the standard library."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise HTTPError(400, "malformed multipart body")
    form, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        if part.get_filename() is not None or name == "audio":
            files[name] = data
        else:
            form[name] = data.decode(part.get_content_charset() or "utf-8", "replace")
    return form, files


def make_handler(app: App):
    from . import __version__

    class Handler(BaseHTTPRequestHandler):
        server_version = f"pronunciada/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):     # request line only; never headers (keys) or bodies
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> bytes:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                out = bytearray()
                while True:
                    line = self.rfile.readline(65537)
                    try:
                        n = int(line.split(b";")[0].strip() or b"0", 16)
                    except ValueError:
                        raise HTTPError(400, "bad chunked encoding") from None
                    if n == 0:
                        while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                            pass
                        return bytes(out)
                    if len(out) + n > app.max_bytes:
                        self.close_connection = True
                        raise HTTPError(413, f"request body larger than {app.max_bytes} bytes")
                    out += self.rfile.read(n)
                    self.rfile.readline(3)
            n = self.headers.get("Content-Length")
            if n is None:
                raise HTTPError(411, "Content-Length or chunked transfer encoding required")
            try:
                n = int(n)
            except ValueError:
                raise HTTPError(400, "bad Content-Length") from None
            if n > app.max_bytes:
                self.close_connection = True
                if n <= 4 * app.max_bytes:         # drain a moderately large body so the client sees the 413
                    left = n
                    while left > 0:
                        got = self.rfile.read(min(left, 1 << 20))
                        if not got:
                            break
                        left -= len(got)
                raise HTTPError(413, f"request body larger than {app.max_bytes} bytes")
            return self.rfile.read(n) if n > 0 else b""

        def _route(self, method: str):
            u = urlsplit(self.path)
            q = parse_qs(u.query, keep_blank_values=True)
            path = u.path.rstrip("/") or "/"
            if path == "/healthz":
                if method != "GET":
                    raise HTTPError(405, "use GET")
                return 200, {"status": "ok", "version": __version__, "azure_compat": app.azure_compat}
            if path == "/v1/languages":
                if method != "GET":
                    raise HTTPError(405, "use GET")
                from .languages import languages

                return 200, {"languages": languages()}
            if path == "/v1/score":
                if method != "POST":
                    raise HTTPError(405, "use POST")
                body = self._body()
                return 200, app.native(body, self.headers.get("Content-Type") or "", q)
            if app.azure_compat and path in AZURE_PATHS:
                if method != "POST":
                    raise HTTPError(405, "use POST")
                body = self._body()
                return 200, app.azure(body, self.headers, q)
            raise HTTPError(404, f"no route {u.path}")

        def _handle(self, method: str):
            try:
                status, obj = self._route(method)
            except HTTPError as e:
                status, obj = e.status, {"error": e.message}
            except Exception as e:             # noqa: BLE001
                status, obj = 500, {"error": f"internal error ({type(e).__name__})"}
            if status >= 400 and method == "POST" and not self.close_connection:
                self.close_connection = True   # unread body may remain; do not reuse the connection
            self._send(status, obj)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_PUT(self):
            self._handle("PUT")

        def do_DELETE(self):
            self._handle("DELETE")

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8765, **app_kw) -> ThreadingHTTPServer:
    app = App(**app_kw)
    if app.device == "cuda":
        app.scorer("en")  # Initialize the requested provider before opening the port.
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8765, **app_kw) -> int:
    httpd = make_server(host, port, **app_kw)
    h, p = httpd.server_address[:2]
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {host}; this server has no authentication", file=sys.stderr)
    extra = ", Azure-format path enabled" if app_kw.get("azure_compat") else ""
    print(f"pronunciada serving on http://{h}:{p} (POST /v1/score{extra}); Ctrl-C to stop", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
