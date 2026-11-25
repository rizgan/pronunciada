"""Layer 1 readings: a reading (kana for Japanese, pinyin for Chinese) -> the target that
pronunciada.core.score_phones takes, by rules only. No dictionary, no third-party package, no data
file; nothing here imports a layer-2 text front-end (pronunciada.text.<lang>). See docs/TEXT_LAYERS.md.
"""
from __future__ import annotations

__all__ = ["phones_from_kana", "phones_from_pinyin"]


def phones_from_kana(reading, text: str | None = None, *, style: str = "pron", devoicing: str = "auto") -> dict:
    """Japanese target (pronunciada.core format, a JSON-able dict) from a kana reading.
    See pronunciada.reading.ja for the input format, the two styles and the devoicing options."""
    from ..core import target_from_reference
    from .ja import reference_from_kana

    return target_from_reference(reference_from_kana(reading, text, style=style, devoicing=devoicing))


def phones_from_pinyin(reading, text: str | None = None, *, sandhi: bool = True) -> dict:
    """Mandarin target (pronunciada.core format) from pinyin. See pronunciada.reading.zh."""
    from ..core import target_from_reference
    from .zh import reference_from_pinyin

    return target_from_reference(reference_from_pinyin(reading, text, sandhi=sandhi))
