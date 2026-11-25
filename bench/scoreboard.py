"""results/SCOREBOARD.md: every held-out comparison on one page, generated from the JSON tables.

    python -m bench.scoreboard            # reads results/tables/*.json, writes results/SCOREBOARD.md

Nothing is computed from audio here; every number is read from a table written by the benchmark
scripts (listed next to each section). The tally counts every comparison with Azure listed on the
page, classified by the table's own paired test (two-sided p < 0.05, or a paired 95 % CI that
excludes 0 where the table gives a CI and no p).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
NAMES = {"en": "English", "zh": "Mandarin", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German",
         "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ko": "Korean"}
CATS = ("ahead (significant)", "ahead (not significant)", "level", "behind (not significant)",
        "behind (significant)")
TALLY: list[dict] = []


def load(name):
    p = T / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}"


def pct(x, d=0):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f} %"


def pv(p):
    if p is None:
        return "–"
    return "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def classify(delta, p=None, ci=None, higher_is_better=True, level_tol=0.0005):
    """Ahead / behind from ours' point of view; significance from the table's own test."""
    d = delta if higher_is_better else -delta
    if abs(d) < level_tol:
        return "level"
    if p is not None:
        sig = p < 0.05
    elif ci is not None:
        lo, hi = ci
        sig = lo > 0 or hi < 0
    else:
        sig = False
    return f"{'ahead' if d > 0 else 'behind'} ({'significant' if sig else 'not significant'})"


def tally(area, lang, measure, delta, p=None, ci=None, higher_is_better=True, level_tol=0.0005):
    c = classify(delta, p, ci, higher_is_better, level_tol)
    TALLY.append({"area": area, "lang": lang, "measure": measure, "cat": c})
    return c


def short(c):
    return {"ahead (significant)": "**ahead**", "ahead (not significant)": "ahead, n.s.", "level": "level",
            "behind (not significant)": "behind, n.s.", "behind (significant)": "**behind**"}[c]


# --------------------------------------------------------------------------- sections

def english(L):
    E = load("english_vs_azure.json")
    if not E:
        return
    s = E["so762"]
    L += ["## English", "",
          "### Agreement with expert raters (speechocean762 test, 2,500 recordings of 125 Mandarin-L1 learners)", "",
          "Pearson correlation with the human scores; 95 % speaker-bootstrap CI; Δ = ours − Azure with a paired "
          "speaker bootstrap (two-sided p). Source: `bench/eval_english.py` → `english_vs_azure.json`.", "",
          "| level | Pronunciada [95 % CI] | Azure [95 % CI] | Δ [95 % CI] | p | verdict |", "|---|---|---|---|---|---|"]
    rows = [("individual sounds", s["phone"]["ours_B"], s["phone"]["azure"]),
            ("words", s["word"]["ours"], s["word"]["azure"])]
    rows += [(f"sentence {k}" + (" (intonation)" if k == "prosodic" else ""), s["sentence"][k]["ours"],
              s["sentence"][k]["azure"]) for k in ("accuracy", "fluency", "prosodic", "total")]
    for name, o, a in rows:
        d = o["minus_azure"]
        c = tally("speechocean762", "en", name, d["delta"], p=d["p_value_two_sided"])
        L.append(f"| {name} | {f3(o['pearson'])} [{f3(o['pearson_ci'][0])}, {f3(o['pearson_ci'][1])}] | "
                 f"{f3(a['pearson'])} [{f3(a['pearson_ci'][0])}, {f3(a['pearson_ci'][1])}] | {d['delta']:+.3f} "
                 f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {pv(d['p_value_two_sided'])} | {short(c)} |")
    L += ["", "### Finding real learner mistakes (expert-marked wrong sounds, test splits)", "",
          "Area under the ROC curve of the sound score against the annotators' marks (higher is better). "
          "Source: `english_vs_azure.json` (`errdet`).", "",
          "| data | Pronunciada [95 % CI] | Azure [95 % CI] | Δ [95 % CI] | p | verdict |", "|---|---|---|---|---|---|"]
    for ds, label in (("l2arctic", "L2-ARCTIC test (24 speakers, 6 held out; 6 L1s)"),
                      ("epadb", "EpaDB test (Spanish-L1 speakers)")):
        a = E["errdet"][ds]["all"]
        o, z = a["ours_B"], a["azure"]
        d = o.get("minus_azure")
        if d:
            c = tally("learner errors", "en", label.split(" (")[0], d["delta"], p=d["p_value_two_sided"])
            dd, pp, vv = f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]", pv(d["p_value_two_sided"]), short(c)
        else:
            dd = pp = vv = "–"
        L.append(f"| {label} | {f3(o['auc'])} [{f3(o['auc_ci'][0])}, {f3(o['auc_ci'][1])}] | {f3(z['auc'])} "
                 f"[{f3(z['auc_ci'][0])}, {f3(z['auc_ci'][1])}] | {dd} | {pp} | {vv} |")
    Y = load("coaching_azure_yardstick.json")
    if Y:
        L += ["", "### Coaching: which sounds to flag (speechocean762 test)", "",
              f"{Y['n_phones']:,} expert-scored sounds, {pct(Y['error_rate'], 1)} marked as errors (expert score < 2). "
              "Ours = the coaching error score at its three sensitivities; Azure = 100 − its phoneme accuracy, "
              "thresholded to reach the same recall. Paired speaker-bootstrap CI of the difference. Source: "
              "`bench/coach/azure_yardstick.py` → `coaching_azure_yardstick.json`.", "",
              "| sensitivity | recall | precision, Pronunciada [CI] | precision, Azure [CI] | Δ CI | verdict |",
              "|---|---|---|---|---|---|"]
        for k in ("careful", "balanced", "eager"):
            r = Y["at_recall"][k]
            c = tally("coaching", "en", f"precision at '{k}' recall", r["precision_ours"] - r["precision_azure"],
                      ci=r["diff_ci"])
            L.append(f"| {k} | {f3(r['recall'])} | {f3(r['precision_ours'])} [{f3(r['ours_ci'][0])}, {f3(r['ours_ci'][1])}] | "
                     f"{f3(r['precision_azure'])} [{f3(r['azure_ci'][0])}, {f3(r['azure_ci'][1])}] | "
                     f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] | {short(c)} |")
        pr = Y["pr_auc"]
        c = tally("coaching", "en", "precision-recall AUC", pr["ours"] - pr["azure"], ci=Y["diff_ap_ci"])
        L.append(f"| precision–recall AUC | | {f3(pr['ours'])} [{f3(pr['ours_ci'][0])}, {f3(pr['ours_ci'][1])}] | "
                 f"{f3(pr['azure'])} [{f3(pr['azure_ci'][0])}, {f3(pr['azure_ci'][1])}] | "
                 f"[{Y['diff_ap_ci'][0]:+.3f}, {Y['diff_ap_ci'][1]:+.3f}] | {short(c)} |")
        h = Y.get("heard", {})
        R4 = load("coaching_r4_heldout.json")
        if R4:
            b = R4["test"]["balanced"]
            L += ["", f"Feature-level tips (no Azure equivalent): at \"balanced\", {pct(b.get('with_claim', b.get('share_with_claim')), 1)} "
                  f"of the {R4['n_substitutions']} real substitutions get a feature claim, right {pct(b.get('precision', b.get('claim_precision')), 0)} "
                  f"of the time; {pct(b.get('any_right_feature'), 1)} get at least one right feature (claim or named sound). "
                  "Source: `bench/coach/fit_r4.py eval` → `coaching_r4_heldout.json`."]
        if h:
            o = h.get("ours_balanced") or {}
            L += ["", f"Naming the sound heard instead, on {h.get('n_substitutions')} real substitutions: ours (balanced) names "
                  f"{pct(o.get('coverage'), 0)} of them, {pct(o.get('within1'), 0)} right within one articulatory feature; "
                  f"Azure's top n-best candidate differs from the expected sound on {pct(h['azure_top_names_other']['coverage'], 0)}, "
                  f"{pct(h['azure_top_names_other']['within1'], 0)} right within one feature."]
    L.append("")


def mandarin(L):
    O, O0 = load("ompal.json"), load("ompal_tone_zh_v2.json")
    if not O:
        return
    L += ["## Mandarin", "",
          "### Real learners: OMPAL test (French-L1 learners of Taiwan Mandarin; 21 speakers, teachers mark every character)", "",
          "Word level: AUC of the word score for characters the teachers marked wrong (all test speakers). Sentence level: "
          "Pearson with the mean teacher score, learners only. Azure zh-TW with the Traditional prompt (primary). "
          "Paired speaker bootstrap, two-sided p. **Current** = the shipped system, whose Mandarin defaults (tone model, "
          "sentence recalibration, word contrast factor) were chosen on OMPAL **dev** learners (different speakers, same L1); "
          "**first contact** = the round-6 system, before anything had looked at OMPAL (historical table "
          "`ompal_tone_zh_v2.json`, not re-run). Azure is untuned in both. Source: `bench/eval_ompal.py` → `ompal.json`.", "",
          "| measure | first contact | current [95 % CI] | Azure [95 % CI] | Δ current − Azure | p | verdict |",
          "|---|---|---|---|---|---|---|"]
    rows = [("word: any error", "word", "primary: any error"), ("word: tone-only error", "word", "(a) tone-only error"),
            ("word: consonant / vowel error", "word", "(b) consonant or vowel error"),
            ("sentence accuracy", "sentence", "accuracy"), ("sentence fluency", "sentence", "fluency"),
            ("sentence prosody", "sentence", "prosody")]
    for name, sec, key in rows:
        def get(R):
            x = R["test"][sec].get(key) if R else None
            if x is None:
                return None
            return x["ours"] if sec == "word" else x["pearson"]
        o, o0 = get(O), get(O0)
        if o is None:
            continue
        c = tally("OMPAL", "zh", name, o["delta"], p=o.get("p_speaker"))
        L.append(f"| {name} | {f3(o0['value']) if o0 else '–'} | {f3(o['value'])} [{f3(o['ci_speaker'][0])}, {f3(o['ci_speaker'][1])}] | "
                 f"{f3(o['azure'])} [{f3(o['azure_ci_speaker'][0])}, {f3(o['azure_ci_speaker'][1])}] | {o['delta']:+.3f} | "
                 f"{pv(o.get('p_speaker'))} | {short(c)} |")
    L.append("")
    S = load("swap_ours_vs_azure.json")
    P = (S or {}).get("paired_ours_minus_azure", {}).get("zh", {})
    if P.get("tone:auc"):
        L += ["### Tones: same syllable, different tone in the prompt (FLEURS test, native audio)", "",
              "| measure | Pronunciada | Azure | Δ [95 % CI] | p | verdict |", "|---|---|---|---|---|---|"]
        for k, name in (("tone:auc", "detection (AUC)"), ("tone:hit@5%FA", "caught at 5 % native false alarms"),
                        ("tone:right_word", "right word blamed")):
            x = P.get(k)
            if not x:
                continue
            c = tally("tone swaps", "zh", name, x["delta"], p=x["p_two_sided"])
            L.append(f"| {name} | {f3(x['a'])} | {f3(x['b'])} | {x['delta']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] | "
                     f"{pv(x['p_two_sided'])} | {short(c)} |")
        L.append("")


def swaps(L):
    S = load("swap_ours_vs_azure.json")
    if not S:
        return
    P = S.get("paired_ours_minus_azure", {})
    L += ["## All ten languages", "",
          "### Swap benchmark: native recording, one word of the prompt changed (FLEURS test, 150 recordings per language)", "",
          "\"far\" = a clearly different word, \"near\" = a word one sound away. AUC = the changed word vs the untouched "
          "words; hit@5 % = share of changed words flagged at the threshold that flags 5 % of the words of the unchanged "
          "recordings; right word = the changed word gets the lowest score; native FA = share of native words below each "
          "system's natural cut-off (0.6 / 60), lower is better (each system at its own cut-off, so this column is "
          "not a matched comparison; the hit@5 % column is). Cells: Pronunciada / Azure; the verdict is the paired "
          "utterance bootstrap on the records both systems scored (two-sided p in brackets). Source: "
          "`bench/eval_languages.py --steps swap` → `swap_ours_vs_azure.json`.", "",
          "| lang | far AUC | near AUC | near hit@5 % | far right word | native FA (< 0.6) |", "|---|---|---|---|---|---|"]
    keys = (("swap_far:auc", "far AUC", True), ("swap_near:auc", "near AUC", True),
            ("swap_near:hit@5%FA", "near hit@5 %", True), ("swap_far:right_word", "far right word", True),
            ("native:fa<60", "native false alarms", False))
    for lang in LANGS:
        p = P.get(lang)
        if not p:
            continue
        cells = []
        for k, name, hib in keys:
            x = p.get(k)
            if not x:
                cells.append("–")
                continue
            c = tally("swap benchmark", lang, name, x["delta"], p=x["p_two_sided"], higher_is_better=hib)
            cells.append(f"{f3(x['a'])} / {f3(x['b'])} {short(c)} ({pv(x['p_two_sided'])})")
        L.append(f"| {lang} | " + " | ".join(cells) + " |")
    L.append("")


def cvreal(L):
    C = load("cv_real.json")
    if not C:
        return
    L += ["### cv-real: real Common Voice recordings judged by volunteers (test half)", "",
          "Test A: clips the Common Voice reviewers rejected vs clips they accepted; Test B: self-declared non-native vs "
          "native speakers (accepted clips, en and de only). AUC of the sentence score (ours: sentence probability; "
          "Azure: PronScore); paired speaker bootstrap, two-sided p. Source: `bench/eval_cv.py` → `cv_real.json`.", "",
          "| locale / test | n (bad / good) | Pronunciada [CI] | Azure [CI] | Δ [CI] | p | verdict |",
          "|---|---|---|---|---|---|---|"]
    for k, r in C["results"].items():
        if k.endswith("Aweak"):
            continue
        p = r["primary"]
        lang = k.split("/")[0].split("-")[0]
        c = tally(f"cv-real test {k.split('/')[1]}", lang, k, p["delta"], p=p["p_two_sided"])
        L.append(f"| {k} | {r['n_pos']} / {r['n_neg']} | {f3(p['ours'])} [{f3(p['ours_ci'][0])}, {f3(p['ours_ci'][1])}] | "
                 f"{f3(p['azure'])} [{f3(p['azure_ci'][0])}, {f3(p['azure_ci'][1])}] | {p['delta']:+.3f} "
                 f"[{p['delta_ci'][0]:+.3f}, {p['delta_ci'][1]:+.3f}] | {pv(p['p_two_sided'])} | {short(c)} |")
    L.append("")


def native(L):
    G = load("languages.json")
    if not G:
        return
    L += ["### Native speakers are not punished (FLEURS test, default strictness 0.5; no Azure column)", "",
          "Share of native sounds / words / sentences accepted (probability ≥ 0.5), and detection of planted sound errors "
          "(AUC; the reference altered, audio native). Source: `bench/eval_languages.py` → `languages.json`.", "",
          "| lang | sounds accepted | words accepted | sentences accepted | planted sound errors, AUC | planted word errors, AUC |",
          "|---|---|---|---|---|---|"]
    for lang in LANGS:
        b = G["by_lang"].get(lang)
        if not b:
            continue
        n = b["native"]["0.5"]
        L.append(f"| {lang} | {pct(n['sounds'], 1)} | {pct(n['words'], 1)} | {pct(n['sentences'], 1)} | "
                 f"{f3(b['sound']['auc'])} | {f3(b['word']['auc'])} |")
    L.append("")


def contrasts(L):
    A = load("contrast_audit_test.json")
    if not A:
        return
    L += ["### Specific sound contrasts (contrast audit, FLEURS test; no Azure column)", "",
          "One sound of the prompt is changed to its partner in a contrast (e.g. plain → aspirated); the native audio "
          "keeps the original. Hit@5 % = share of changed sounds (or their words) flagged at the threshold that flags 5 % "
          "of native sounds (words). Azure was not run on these edits (it gives no comparable per-sound scores outside "
          "English), so there is no head-to-head. 250 test clips per language, the plan rule of the validation audit; "
          "\"blind\" < 30 %, \"weak\" 30–60 %, \"seen\" ≥ 60 % at sound level. Japanese length is judged by its "
          "own channel (next table), so its rows here are omitted. Source: "
          "`bench/audit/contrast_audit.py run/report --split test` → `contrast_audit_test.json`.", "",
          "| lang | contrast | direction (said → expected) | n | sound hit@5 % | word hit@5 % | verdict |",
          "|---|---|---|---|---|---|---|"]
    for lang in LANGS:
        r = A["languages"].get(lang)
        if not r:
            continue
        for cn, c in r["contrasts"].items():
            if not c.get("n") or (lang == "ja" and cn in ("vowel_length", "gemination")):
                continue      # Japanese length is judged by the length channel: next table
            for dn, d in c["directions"].items():
                nm = dn.replace("said ", "").replace(", expected ", " → ")
                L.append(f"| {lang} | {cn} | {nm} | {d['n']} | {pct(d['sound_hit5'])} | {pct(d['word_hit5'])} | "
                         f"{d['verdict']} |")
    L.append("")


def length(L):
    H = load("length_runtime_ja.json")
    if not H:
        return
    L += ["### Japanese vowel length and っ (FLEURS test, planted length swaps; no Azure column)", "",
          f"Shipped runtime (`PhoneResult.length_prob`, `WordResult.prob`), {H['n_clips']} test clips. Caught = at or "
          "below the threshold that flags 5 % of native judged sounds (words). Source: "
          "`bench/length/heldout_runtime.py` → `length_runtime_ja.json`.", "",
          "| direction (said → expected) | n | sound caught | word caught |", "|---|---|---|---|"]
    names = {"vowel_length:A>B": "long vowel → short", "vowel_length:B>A": "short vowel → long",
             "gemination:A>B": "doubled consonant → single", "gemination:B>A": "single consonant → doubled",
             "all": "all"}
    for k, name in names.items():
        if k in H["sound"]:
            L.append(f"| {name} | {H['sound'][k]['n']} | {pct(H['sound'][k]['hit5'])} | {pct(H['word'][k]['hit5'])} |")
    L += ["", f"Native word acceptance at strictness 0.5 on these clips: {pct(H['native_word_accept_0.5'], 1)}.", ""]


def coach_tips(L):
    R = load("coaching_tips_r19.json")
    if not R:
        return
    N = R["natives_fleurs_test"]
    n = next(iter(N.values()))["before"]["clips"]
    L += ["### Coaching tips on native speakers (FLEURS test; no Azure column)", "",
          f"Share of {n} native test clips per language that get at least one pronunciation tip from "
          "`tips(res.coaching(), 3, res)` (\"balanced\", strictness 0.5); every such tip is a false alarm. Before = "
          "every candidate may become a tip; after = round 19, a tip only where the scorer did not accept the "
          "sound or the word. Source: `bench/coach/tips_r19.py` → `coaching_tips_r19.json`.", "",
          "| lang | clips with a pronunciation tip, before → after | tips per clip, before → after |", "|---|---|---|"]
    for lang in LANGS:
        b = N.get(lang)
        if b:
            L.append(f"| {lang} | {pct(b['before']['share_clips_with_pron_tip'])} → "
                     f"{pct(b['after']['share_clips_with_pron_tip'])} | {b['before']['pron_tips_per_clip']:.2f} → "
                     f"{b['after']['pron_tips_per_clip']:.2f} |")
    E, Z = R["speechocean762_test"], R["ompal_test"]
    L += ["", "Learners, before → after: English (speechocean762 test) utterances with an expert-marked wrong sound "
          f"that get a pronunciation tip {pct(E['before']['a_marked_utts_with_pron_tip'])} → "
          f"{pct(E['after']['a_marked_utts_with_pron_tip'])}, tips right {pct(E['before']['b_pron_tips_right'])} → "
          f"{pct(E['after']['b_pron_tips_right'])}, clean utterances with a tip {pct(E['before']['c_clean_utts_with_pron_tip'])} → "
          f"{pct(E['after']['c_clean_utts_with_pron_tip'])}; Mandarin (OMPAL test) "
          f"{pct(Z['before']['a_marked_utts_with_pron_tip'])} → {pct(Z['after']['a_marked_utts_with_pron_tip'])}, "
          f"{pct(Z['before']['b_pron_tips_right'])} → {pct(Z['after']['b_pron_tips_right'])}, "
          f"{pct(Z['before']['c_clean_utts_with_pron_tip'])} → {pct(Z['after']['c_clean_utts_with_pron_tip'])}.", ""]


def speed(L):
    E = load("english_vs_azure.json") or {}
    sp = E.get("speed")
    G = load("languages.json") or {}
    L += ["## Speed, size, and what runs where", ""]
    if sp:
        L += [f"End-to-end `Scorer.score` on 100 speechocean762 clips (mean {sp['fp32_t4']['mean_audio_s']:.1f} s of audio), "
              "laptop CPU (Apple M3 Pro), no other benchmark running (1-minute load average 3.0 at the start, other "
              "processes included), "
              "posterior cache cleared per clip "
              "(`bench/eval_english.py --steps speed`):", "",
              "| model | 4 threads, median (p95) | 1 thread, median (p95) |", "|---|---|---|"]
        for prec in ("fp32", "int8"):
            a, b = sp.get(f"{prec}_t4"), sp.get(f"{prec}_t1")
            if a and b:
                L.append(f"| {prec} | {a['median_ms']:.0f} ms ({a['p95_ms']:.0f}) | {b['median_ms']:.0f} ms ({b['p95_ms']:.0f}) |")
        mb = sp.get("model_mb", {})
        L += ["", f"Acoustic model download on first use: {mb.get('model.onnx', '–')} MB (fp32) or "
              f"{mb.get('model.int8.onnx', '–')} MB (int8, `Scorer(lang, precision=\"int8\")`); MB = 10^6 bytes. "
              "The model cache keeps one file per precision (the download with the extra encoder outputs declared, "
              "the same size), so that is also the size on disk."]
    rows = []
    for lang in LANGS:
        b = (G.get("by_lang") or {}).get(lang)
        a = load(f"azure_swap_{lang}.json")
        if b and a and a.get("latency"):
            la = b["latency"]
            rows.append(f"| {lang} | {la['median_ms']:.0f} ms ({la['median_audio_s']:.1f} s) | "
                        f"{a['latency']['median_s']:.2f} s ({a['latency']['mean_audio_s']:.1f} s) |")
    if rows:
        L += ["", "Longer clips next to Azure (not a controlled comparison: different clip sets of similar length; ours = "
              "FLEURS test, 3 processes with 1 thread each in parallel; Azure = REST round trip to westus incl. upload, "
              "fleurs-swap clips):", "", "| lang | Pronunciada median (median audio) | Azure median (mean audio) |",
              "|---|---|---|"] + rows
    L += ["", "What runs where: everything runs on the device (numpy + ONNX Runtime, no network after the first "
          "model download, no audio leaves the device). Shipped package data ≈ 36 MB (coaching models ≈ 13 MB; "
          "pronunciation dictionaries ≈ 10 MB; adapters, calibrators, tone / stress models ≈ 11 MB; the rest ≈ 2 MB). "
          "No phone build exists yet; nothing has been timed on a phone.", ""]


def how_to_read(L):
    L += ["## How to read this", "",
          "- **Native recordings with altered prompts are not learner speech.** The swap benchmark, planted errors, "
          "the contrast audit and the Japanese length rows take native audio and change the text. They show whether a "
          "system hears that the audio does not match the text, not how it grades real learners; real mistakes are "
          "subtler (English: AUC 0.71–0.72 on real learner errors against 0.98 on the controlled near-swaps).",
          "- **Real-learner evidence exists only for English** (speechocean762, L2-ARCTIC, EpaDB) **and Mandarin** "
          "(OMPAL). The other eight languages have no learner corpus with expert marks here.",
          "- **cv-real labels are volunteers' votes**, not expert pronunciation marks: a clip can be rejected for noise, "
          "a misread word or a bad microphone. Test B uses self-declared accents.",
          "- **Azure was sent each clip once** (cached responses, Nov 2026, region westus, the locales listed in each "
          "results file); its run-to-run variation is not measured. Azure has no per-word stress, no length channel and "
          "no per-sound scores outside English, so those rows have no Azure column.",
          "- **Data the system was tuned on**: speechocean762 train, FLEURS train / validation, OMPAL dev, cv-real dev. "
          "Every row above is on a held-out split. **L2-ARCTIC and EpaDB are fully unseen**: since the license-clean "
          "refit (results/licence_refit.md) no shipped model, threshold or selection used either corpus, train or "
          "test. The current Mandarin system's defaults were chosen on OMPAL dev learners, so on OMPAL test it "
          "is no longer at first contact; the round-6 column is.",
          "- **Design rounds followed earlier held-out looks.** Rounds 2–19 were each designed after an earlier "
          "held-out evaluation had been seen, and each change was selected on separate dev data (train / validation / "
          "dev halves) before the held-out sets were run once. The held-out sets have therefore been looked at many "
          "times across rounds; small differences should be read with that in mind.",
          "- **Significance**: \"significant\" = two-sided paired p < 0.05 in the table's own test (or a paired 95 % CI "
          "excluding 0 where the table has no p); no correction for the number of comparisons. \"Level\" = equal at the "
          "shown precision.", ""]


def zipa_alone(L):
    """Ablation against the unmodified ZIPA model (bench/eval_zipa_baseline.py). Not part of the
    Azure tally."""
    Z = load("zipa_baseline.json")
    if not Z or not Z.get("rows"):
        return
    rows = Z["rows"]
    t = Z.get("tally_vs_gop", {})
    st = Z.get("run", {})

    def cell(x):
        if x.get("by_construction"):
            return "0.5 (blind)"
        s = f3(x["value"])
        return s + (f" [{f3(x['ci'][0])}, {f3(x['ci'][1])}]" if x.get("ci") else "")

    def dcell(r):
        c = r.get("delta_ci")
        return f"{r['delta']:+.3f}" + (f" [{c[0]:+.3f}, {c[1]:+.3f}]" if c else "")

    L += ["## Against ZIPA alone", "",
          "Ablation: the same held-out items, metrics and resampling as the tables below, with Pronunciada replaced by "
          "the unmodified ZIPA model (`zipa-small-crctc-500k`, fp32, raw log-posteriors, no adapter) and the canonical "
          "phone sequence of Pronunciada's text front end (no alternatives, calibrators, prosody / length / contrast "
          "heads or fitted thresholds). **gop** = mean log-posterior of the canonical token over its CTC Viterbi-aligned "
          "frames (word / sentence = mean of their phones); **match** = greedy decode aligned to the canonical tokens "
          "(phone matched or not; sentence = 1 − token error rate). Δ = Pronunciada − gop, paired on the table's own "
          "resampling unit, two-sided p. Pronunciada's values are recomputed from the cached per-utterance outputs and "
          f"equal the tables below. Baseline audio: {st.get('unique_audio', '–')} files, "
          f"{st.get('audio_hours', float('nan')):.1f} h. Source: `bench/eval_zipa_baseline.py` → `zipa_baseline.json` "
          "(write-up: results/zipa_baseline.md). These rows are not counted in the Azure tally.", "",
          f"Pronunciada vs gop over {t.get('n', len(rows))} rows: significantly ahead {t.get('ahead (significant)', '–')}, "
          f"not different {t.get('not different', '–')}, significantly behind {t.get('behind (significant)', '–')}.", ""]
    head = ["| test | n | Pronunciada [95 % CI] | ZIPA gop [95 % CI] | ZIPA match [95 % CI] | Δ vs gop [95 % CI] | p | verdict |",
            "|---|---|---|---|---|---|---|---|"]
    titles = {"speechocean762": "speechocean762 test, Pearson with the expert scores (speaker bootstrap)",
              "learner errors": "L2-ARCTIC and EpaDB test, ROC AUC of the sound score against the annotators' marks "
                                "(speaker bootstrap)",
              "coaching": "speechocean762 test coaching yardstick, precision at Pronunciada's three recalls (gop and "
                          "match thresholded to the same recall; speaker bootstrap)",
              "OMPAL": "OMPAL test, word AUC (all test speakers) and sentence-accuracy Pearson (learners); speaker "
                       "bootstrap",
              "tone swaps": "Tone swaps (zh, FLEURS test), ROC AUC; utterance bootstrap",
              "cv-real test A": "cv-real test A, AUC of the sentence score (speaker bootstrap)",
              "cv-real test B": "cv-real test B, AUC of the sentence score (speaker bootstrap)"}
    for g in ("speechocean762", "learner errors", "coaching", "OMPAL", "tone swaps", "swap benchmark",
              "cv-real test A", "cv-real test B"):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        if g == "swap benchmark":
            L += ["**Swap benchmark (FLEURS test, 10 languages; utterance bootstrap).** Cells: Pronunciada / gop / match, "
                  "then Δ vs gop and verdict (p).", "",
                  "| lang | far AUC | near AUC | near hit@5 % |", "|---|---|---|---|"]
            for lang in LANGS:
                cells = []
                for k in ("far AUC", "near AUC", "near hit@5 %"):
                    r = next((x for x in rs if x["test"] == f"{lang} {k}"), None)
                    cells.append("–" if r is None else
                                 f"{f3(r['pronunciada']['value'])} / {f3(r['gop']['value'])} / {f3(r['match']['value'])}, "
                                 f"{r['delta']:+.3f} {short(r['verdict'])} ({pv(r['p'])})")
                if any(c != "–" for c in cells):
                    L.append(f"| {lang} | " + " | ".join(cells) + " |")
            L.append("")
            continue
        L += [f"**{titles[g]}.**", ""] + head
        for r in rs:
            L.append(f"| {r['test']} | {r['n']} | {cell(r['pronunciada'])} | {cell(r['gop'])} | {cell(r['match'])} | "
                     f"{dcell(r)} | {pv(r['p'])} | {short(r['verdict'])} |")
        note = next((r.get("note") for r in rs if g == "tone swaps" and r.get("note")), None)
        L += ([""] + [note] if note else []) + [""]
    rk = Z.get("rank_check") or []
    if rk:
        L += ["**Rank check (Spearman).** GOP scores are unbounded log-posteriors with a floor, so Pearson could "
              "understate the baseline through nonlinearity alone; this repeats the Pearson rows with Spearman on the "
              "same items and the same paired speaker bootstrap (not counted in the row count above).", "",
              "| test | n | Pronunciada [95 % CI] | ZIPA gop [95 % CI] | ZIPA match [95 % CI] | Δ vs gop [95 % CI] | p | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rk:
            L.append(f"| {r['group']} {r['test']} | {r['n']} | {cell(r['pronunciada'])} | {cell(r['gop'])} | "
                     f"{cell(r['match'])} | {dcell(r)} | {pv(r['p'])} | {short(r['verdict'])} |")
        L.append("")


def tally_section(L):
    L += ["## Tally against Azure", "",
          f"Every comparison with an Azure column above, {len(TALLY)} in total, each classified by its own table's paired "
          "test.", "", "| area | n | " + " | ".join(CATS) + " |", "|---|---|" + "---|" * len(CATS)]
    by = defaultdict(lambda: defaultdict(int))
    for t in TALLY:
        by[t["area"]][t["cat"]] += 1
        by["all"][t["cat"]] += 1
    for area in [a for a in dict.fromkeys(t["area"] for t in TALLY)] + ["all"]:
        n = sum(by[area].values())
        name = f"**{area}**" if area == "all" else area
        L.append(f"| {name} | {n} | " + " | ".join(str(by[area][c]) for c in CATS) + " |")
    L += ["", "By language:", "", "| lang | n | " + " | ".join(CATS) + " |", "|---|---|" + "---|" * len(CATS)]
    bl = defaultdict(lambda: defaultdict(int))
    for t in TALLY:
        bl[t["lang"]][t["cat"]] += 1
    for lang in LANGS:
        if lang in bl:
            L.append(f"| {lang} | {sum(bl[lang].values())} | " + " | ".join(str(bl[lang][c]) for c in CATS) + " |")
    behind = [t for t in TALLY if t["cat"] == "behind (significant)"]
    L += ["", "Significantly behind: " + ("; ".join(f"{t['lang']} {t['area']}: {t['measure']}" for t in behind)
                                          if behind else "none") + ".", ""]


def main():
    body: list[str] = []
    english(body)
    mandarin(body)
    swaps(body)
    cvreal(body)
    native(body)
    contrasts(body)
    length(body)
    coach_tips(body)
    speed(body)
    head = ["# Pronunciada scoreboard", "",
            "Every held-out number in one place, for the code on `main`: the round-20 code (commit fce7918) merged with "
            "the learner benchmark, the size / CPU-speed work and the two-layer text front end (own number speller, "
            "kana and pinyin input, one backbone file per precision in the model cache, no second Japanese "
            "dictionary). The only scoring change in that merge is the removal of the second Japanese dictionary "
            "(Sudachi): its Japanese rows come from one held-out run of that code on 2026-11-23 (18 table rows moved, "
            "all Japanese, none failing the quality gate); every other language gives the outputs of the round-20 code (the "
            "scoring targets of all 40,585 benchmark prompts are identical outside Japanese, and the backbone outputs "
            "are identical under the new file layout). The round-20 code is round 19 plus the round-20 Mandarin word step "
            "and Korean / Japanese speed-up: the round-19 frozen run of 2026-11-22 (results/round19_integration.md) with "
            "every Mandarin row re-run on that code the same day; the other nine languages give bitwise-identical outputs "
            "under it (results/round20_release_check.md). Plus the round-20 Spanish false-alarm fixes (number readings, "
            "glides), whose Spanish rows come from one held-out run on 2026-11-23; every other language is bitwise "
            "identical to that code (results/round20_fa_release_check.md). Scores are the fp32 Scorer's. Next to "
            "Microsoft Azure Pronunciation Assessment where a comparison exists. Generated by `python -m bench.scoreboard` "
            "from `results/tables/*.json`; do not edit by hand. Commands to regenerate each table: README "
            "\"Reproducing the benchmarks\".", ""]
    tl: list[str] = []
    tally_section(tl)
    hr: list[str] = []
    how_to_read(hr)
    za: list[str] = []
    zipa_alone(za)          # reads zipa_baseline.json only; adds nothing to TALLY
    out = head + tl + hr + za + body
    (ROOT / "results" / "SCOREBOARD.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (T / "scoreboard_tally.json").write_text(json.dumps(TALLY, indent=1, ensure_ascii=False))
    print(f"wrote results/SCOREBOARD.md ({len(TALLY)} comparisons)")


if __name__ == "__main__":
    main()
