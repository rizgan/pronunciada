"""pronunciada: on-device multilingual pronunciation scoring built on ZIPA.

    import pronunciada
    scorer = pronunciada.Scorer("en")
    res = scorer.score("clip.wav", "The quick brown fox", strictness=0.5)
    pronunciada.score("clip.wav", "The quick brown fox", "en")
"""
__version__ = "1.1.0"

from .types import (PhoneResult, Reference, SentenceResult, SyllableResult,  # noqa: E402,F401
                    WordResult)


def score(audio, text, lang: str = "en", strictness: float = 0.5, **kw) -> SentenceResult:
    """One call: score `audio` (path | bytes | (array, rate) | 16 kHz array) against `text`.
    A Scorer is built once per (language, options) and reused; `kw` goes to Scorer()."""
    from .scorer import score as _score
    from .text import normalize_lang

    return _score(audio, text, normalize_lang(lang), strictness=strictness, **kw)


def __getattr__(name):
    # lazy: importing pronunciada stays cheap (no onnxruntime / model load until a Scorer is built)
    if name == "Scorer":
        from . import scorer

        return getattr(scorer, name)
    if name in ("score_phones", "phones_for", "phones_from_kana", "phones_from_pinyin"):  # pronunciada.core
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module 'pronunciada' has no attribute {name!r}")


__all__ = ["Scorer", "score", "score_phones", "phones_for", "phones_from_kana", "phones_from_pinyin",
           "SentenceResult", "WordResult", "SyllableResult",
           "PhoneResult",
           "Reference", "__version__"]
