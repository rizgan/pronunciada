from types import SimpleNamespace

from bench.coach.en_precision import evaluate_record, summarize


def test_measure_public_marks_by_position(monkeypatch):
    from bench.coach import en_precision as module

    ref = SimpleNamespace(
        words=[SimpleNamespace(text="MAMA", phones=(0, 4)),
               SimpleNamespace(text="MAMA", phones=(4, 8))],
        phones=[SimpleNamespace(ipa=ipa, alts=()) for ipa in ("m", "ɑ") * 4],
    )
    positions = {"word": [0] * 4 + [1] * 4, "k": list(range(4)) * 2,
                 "expected": ["m", "ɑ"] * 4}
    items = [
        {"kind": "substitution", "word_index": 0, "phone_index": 2},
        {"kind": "missing", "word_index": 0, "phone_index": 2},
        {"kind": "missing", "word_index": 0, "phone_index": 3},
        {"kind": "missing_word", "word_index": 1, "phone_index": None},
    ]
    monkeypatch.setattr(module.coach, "sound_inputs", lambda result: positions)
    monkeypatch.setattr(module.coach, "flagged_indices", lambda *args: [2, 3, 4])
    monkeypatch.setattr(module.coach, "error_prob", lambda *args, **kwargs: [0.5] * 8)

    class Result:
        _pronunciada = {"ref": ref}

        def coaching(self, *, sensitivity):
            assert sensitivity == "balanced"
            return items

    record = {"id": "test-record", "speaker": 1, "text": "MAMA MAMA",
              "labels": {"words": [
                  {"text": "MAMA", "phones": ["M", "AA", "M", "AA"],
                   "phones_accuracy": [2, 2, 1, None]},
                  {"text": "MAMA", "phones": ["M", "AA", "M", "AA"],
                   "phones_accuracy": [1, 2, 2, 2]},
              ]}}
    measured = evaluate_record(record, "sample.flac", Result())
    assert measured["phones"][0]["expert_position"] == (0, 0)
    assert not measured["phones"][0]["marked"]
    assert measured["phones"][2]["expert_position"] == (0, 2)
    assert measured["phones"][4]["expert_position"] == (1, 0)
    assert measured["phones"][4]["omitted_word"]
    assert measured["expert_losses"] == [
        {"expert_position": (0, 3), "reason": "missing_expert_score", "arpa": "AA"}
    ]
    report = summarize([measured])
    assert report["metrics"]["items"] == {
        "TP": 1, "FP": 0, "FN": 1, "TN": 5, "precision": 1.0, "recall": 0.5,
    }
    assert report["metrics"]["raw"]["TP"] == 2
    assert report["coverage"]["items_without_expert_label"] == 1
    assert report["coverage"]["matched_in_omitted_words"] == 4
