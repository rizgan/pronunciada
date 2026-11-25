"""Pronunciada demo for Hugging Face Spaces: score a recording against its reference text."""
from __future__ import annotations

import html
import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

import pronunciada
from pronunciada.scorer import Scorer

HERE = Path(__file__).parent
LANGS = {
    "English": "en", "中文 (Mandarin)": "zh", "日本語 (Japanese)": "ja", "Español (Spanish)": "es",
    "Français (French)": "fr", "Deutsch (German)": "de", "Italiano (Italian)": "it",
    "Português (Portuguese)": "pt", "Русский (Russian)": "ru", "한국어 (Korean)": "ko",
}
NO_SPACES = {"zh", "ja"}
SENSITIVITY = {"Careful (fewer tips)": "careful", "Balanced": "balanced", "Eager (more tips)": "eager"}

CSS = """
.pr-wrap{font-size:15px;line-height:23px}
.pr-sent{font-size:26px;line-height:34px;letter-spacing:-.015em;margin:0 0 8px}
.pr-low{color:#1d4ed8}
.dark .pr-low{color:#7db8ff}
.pr-score{font-size:14px;line-height:20px;opacity:.7;margin:0 0 12px;font-variant-numeric:tabular-nums}
.pr-tip{margin:0 0 6px}
.pr-note{font-size:13px;opacity:.7;margin:12px 0 0}
"""


THEME = gr.themes.Base(
    primary_hue="neutral", neutral_hue="stone", radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#fdfcfc", block_background_fill="#ffffff", block_border_width="1px",
    button_primary_background_fill="#000000", button_primary_background_fill_hover="#262626",
    button_primary_text_color="#ffffff", button_large_radius="9999px", button_small_radius="9999px",
)


@lru_cache(maxsize=3)
def scorer(lang: str) -> Scorer:
    return Scorer(lang, precision=os.environ.get("PRONUNCIADA_PRECISION", "int8"))


def _f(p) -> str:
    return "" if p is None else f"{p:.2f}"


def _low(p) -> bool:
    return p is not None and p < 0.5


def render(res, lang: str, tips: list[dict]) -> str:
    """The sentence with words below 0.5 in red, one line of scores, then the tips."""
    e = html.escape
    sep = "" if lang in NO_SPACES else " "
    words, line, notes = [], [f"Sentence {_f(res.prob)}"], []
    for w in res.words:
        words.append(f'<span class="pr-low">{e(w.text)}</span>' if _low(w.prob) else e(w.text))
        if _low(w.prob):
            line.append(f"{e(w.text)} {_f(w.prob)}")
        for y in w.syllables:
            if getattr(y, "tone_prob", None) is not None and _low(y.tone_prob):
                line.append(f"tone of {e(y.text or '')} {_f(y.tone_prob)}")
                heard = f", tone {y.tone_heard} recognized" if y.tone_heard is not None else ""
                notes.append(f"{e(y.text or '')}: tone {y.tone_expected} expected{heard}")
    out = [f'<div class="pr-wrap"><p class="pr-sent">{sep.join(words)}</p><p class="pr-score">{" · ".join(line)}</p>']
    out += [f'<p class="pr-tip">{e(t["text"])}</p>' for t in tips if t.get("text")]
    if notes:
        out.append(f'<p class="pr-tip">{"; ".join(notes)}.</p>')
    audio = getattr(res, "audio", None)
    issues = audio.get("issues") if isinstance(audio, dict) else None
    if issues:
        out.append(f'<p class="pr-tip">Recording: {e(", ".join(issues))}.</p>')
    out.append('<p class="pr-note">Probabilities that a human rater would accept the sentence and each word. Words below 0.5 are in red. '
               "Phoneme-level scores are in the JSON output.</p></div>")
    return "".join(out)


def assess(audio, text, language, strictness, sensitivity):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    if not (text or "").strip():
        raise gr.Error("Enter the reference text.")
    lang = LANGS[language]
    try:
        res = scorer(lang).score(audio, text.strip(), strictness=float(strictness))
    except ValueError as err:
        raise gr.Error(f"Could not score this: {err}")
    from pronunciada.tips import tips as make_tips

    try:
        items = res.coaching(max_items=6, sensitivity=SENSITIVITY[sensitivity])
    except TypeError:
        items = res.coaching(max_items=6)
    tips = make_tips(items, max_tips=3, result=res)
    try:
        payload = json.loads(res.to_json())
    except Exception:
        payload = {"prob": res.prob, "words": [{"text": w.text, "prob": w.prob} for w in res.words]}
    return render(res, lang, tips), payload


def examples():
    rows = []
    order = ["en-learner-1", "en-learner-2", "zh-learner-1", "en-clean"]
    paths = sorted((HERE / "samples").glob("*.json"), key=lambda q: (order.index(q.stem) if q.stem in order else len(order), q.stem))
    for meta_path in paths:
        meta = json.loads(meta_path.read_text())["meta"]
        name = next((k for k, v in LANGS.items() if v == meta["lang"]), None)
        if name:
            rows.append([str(meta_path.with_suffix(".mp3")), meta["text"], name, 0.5, "Balanced"])
    return rows


with gr.Blocks(title="Pronunciada") as demo:
    gr.Markdown(
        "# Pronunciada\n"
        "Pronunciation assessment for ten languages. Record or upload a sentence, enter the text that was read, "
        "and the demo returns the probability that each word and phoneme would be accepted by a human rater. "
        "[Website](https://pronunciada.dev) · [Source](https://github.com/moglearn/pronunciada) · "
        "[Benchmarks](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md)"
    )
    with gr.Row():
        with gr.Column(scale=5):
            language = gr.Dropdown(list(LANGS), value="English", label="Language")
            text = gr.Textbox(label="Reference text", placeholder="The sentence that is read in the recording", lines=2)
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recording",
                             waveform_options=gr.WaveformOptions(waveform_color="#b9b4ae", waveform_progress_color="#000000"))
            with gr.Accordion("Settings", open=False):
                strictness = gr.Slider(0, 1, value=0.5, step=0.05, label="Strictness (0 lenient, 1 strict)")
                sensitivity = gr.Radio(list(SENSITIVITY), value="Balanced", label="Coaching tips")
            go = gr.Button("Score", variant="primary")
        with gr.Column(scale=6):
            report = gr.HTML(label="Result")
            with gr.Accordion("Full JSON output", open=False):
                raw = gr.JSON()
    rows = examples()
    if rows:
        gr.Examples(rows, inputs=[audio, text, language, strictness, sensitivity], label="Samples from speechocean762, OMPAL and FLEURS (CC BY 4.0)")
    gr.Markdown(
        "Recordings are scored on this Space's CPU and are not stored. The acoustic model is ZIPA (CC BY 4.0)."
    )
    go.click(assess, [audio, text, language, strictness, sensitivity], [report, raw])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME)
