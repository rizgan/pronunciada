"""bench/opt/gate.py decision logic on synthetic rows, and the row extraction against the scoreboard."""
from pathlib import Path

import pytest

from bench.opt.gate import decide, deep_equal

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"


def row(key="speechocean762|en|sentence total", value=0.7, kind="pcc", hib=True, cat=None, area=None, lang="en"):
    a, l, m = key.split("|")
    return {"key": key, "area": area or a, "lang": lang or l, "measure": m, "value": value, "hib": hib, "kind": kind,
            "cat": cat, "tally": cat is not None}


def paired(vb, vc, p, key="speechocean762|en|sentence total"):
    return {key: {"value_base": vb, "value_cand": vc, "delta": vc - vb, "ci": [vc - vb - 0.01, vc - vb + 0.01],
                  "p": p, "n": 100, "test": "synthetic"}}


def test_identical_rows_pass_and_nothing_moves():
    r = decide([row()], [row()])
    assert r["verdict"] == "PASS" and r["n_moved"] == 0


def test_pcc_drop_of_0_0021_fails():
    r = decide([row(value=0.7)], [row(value=0.6979)])
    assert r["verdict"] == "FAIL"
    assert "Pearson drops" in r["fails"][0]["text"]


def test_pcc_drop_of_0_0019_passes_and_is_listed():
    r = decide([row(value=0.7)], [row(value=0.6981)])
    assert r["verdict"] == "PASS"
    assert [m["key"] for m in r["moved"]] == ["speechocean762|en|sentence total"]
    assert r["moved"][0]["worse"] and not r["moved"][0]["fails"]


def test_in_tolerance_move_with_nonsignificant_paired_test_passes_and_is_listed():
    r = decide([row(value=0.7)], [row(value=0.6995)], paired(0.7, 0.6995, p=0.4))
    assert r["verdict"] == "PASS" and r["n_moved"] == 1 and r["moved"][0]["p"] == 0.4


def test_significantly_worse_paired_fails_even_inside_tolerance():
    r = decide([row(value=0.7)], [row(value=0.6995)], paired(0.7, 0.6995, p=0.01))
    assert r["verdict"] == "FAIL" and "significantly worse" in r["fails"][0]["text"]


def test_budget_rule_lists_significance_inside_tolerance():
    r = decide([row(value=0.7)], [row(value=0.6995)], paired(0.7, 0.6995, p=0.01))
    assert r["verdict"] == "FAIL" and r["verdict_budget"] == "PASS"
    assert len(r["budget_listed"]) == 1 and not r["budget_fails"]
    assert r["chance"]["rows_with_paired_p"] == 1 and r["chance"]["observed_worse_and_p_below_0.05"] == 1


def test_budget_rule_still_fails_on_tolerances_and_classes():
    r = decide([row(value=0.7)], [row(value=0.6979)], paired(0.7, 0.6979, p=0.5))
    assert r["verdict"] == r["verdict_budget"] == "FAIL"
    k = "swap benchmark|de|near AUC"
    r = decide([row(k, 0.98, "detection_auc", cat="ahead (significant)")],
               [row(k, 0.9799, "detection_auc", cat="ahead (not significant)")])
    assert r["verdict_budget"] == "FAIL"
    k = "length|ja|all word caught"
    r = decide([row(k, 0.50, "rate")], [row(k, 0.499, "rate")])        # no paired test: strict only
    assert r["verdict"] == "FAIL" and r["verdict_budget"] == "PASS"


def test_significantly_better_passes():
    r = decide([row(value=0.7)], [row(value=0.7005)], paired(0.7, 0.7005, p=0.001))
    assert r["verdict"] == "PASS"


def test_flipped_azure_win_fails():
    k = "swap benchmark|de|near AUC"
    b = row(k, 0.98, "detection_auc", cat="ahead (significant)")
    c = row(k, 0.98, "detection_auc", cat="level")
    for rule in ("strict", "coarse"):
        r = decide([b], [c], class_rule=rule)
        assert r["verdict"] == "FAIL", rule
        assert "tally class" in r["fails"][0]["text"]


def test_loss_of_significance_fails_strict_only():
    k = "swap benchmark|de|near AUC"
    b = row(k, 0.98, "detection_auc", cat="ahead (significant)")
    c = row(k, 0.98, "detection_auc", cat="ahead (not significant)")
    assert decide([b], [c], class_rule="strict")["verdict"] == "FAIL"
    assert decide([b], [c], class_rule="coarse")["verdict"] == "PASS"


def test_class_moving_up_passes():
    k = "swap benchmark|de|near AUC"
    b = row(k, 0.98, "detection_auc", cat="behind (significant)")
    c = row(k, 0.981, "detection_auc", cat="behind (not significant)")
    assert decide([b], [c])["verdict"] == "PASS"


def test_auc_and_native_false_alarm_tolerances():
    k = "planted errors|it|word errors, AUC"
    assert decide([row(k, 0.95, "detection_auc")], [row(k, 0.9479, "detection_auc")])["verdict"] == "FAIL"
    assert decide([row(k, 0.95, "detection_auc")], [row(k, 0.9481, "detection_auc")])["verdict"] == "PASS"
    k = "swap benchmark|fr|native false alarms"
    assert decide([row(k, 0.030, "native_fa", hib=False)], [row(k, 0.0311, "native_fa", hib=False)])["verdict"] == "FAIL"
    assert decide([row(k, 0.030, "native_fa", hib=False)], [row(k, 0.0309, "native_fa", hib=False)])["verdict"] == "PASS"
    assert decide([row(k, 0.030, "native_fa", hib=False)], [row(k, 0.010, "native_fa", hib=False)])["verdict"] == "PASS"
    k = "natives accepted|ko|words accepted"
    assert decide([row(k, 0.98, "native_accept")], [row(k, 0.9789, "native_accept")])["verdict"] == "FAIL"
    assert decide([row(k, 0.98, "native_accept")], [row(k, 0.9791, "native_accept")])["verdict"] == "PASS"


def test_rate_row_without_paired_test_fails_when_worse_prose_rows_only_listed():
    k = "length|ja|all word caught"
    assert decide([row(k, 0.50, "rate")], [row(k, 0.499, "rate")])["verdict"] == "FAIL"
    assert decide([row(k, 0.50, "rate")], [row(k, 0.501, "rate")])["verdict"] == "PASS"
    k = "coaching (heard)|en|names the sound heard: coverage"
    r = decide([row(k, 0.50, "other")], [row(k, 0.40, "other")])
    assert r["verdict"] == "PASS" and r["n_moved"] == 1 and any("not gating" in n for n in r["notes"])


def test_paired_result_that_does_not_reproduce_the_table_is_not_used():
    k = "length|ja|all word caught"
    r = decide([row(k, 0.50, "rate")], [row(k, 0.499, "rate")], paired(0.51, 0.499, p=0.9, key=k))
    assert r["verdict"] == "FAIL" and r["n_selfcheck_mismatch"] == 1
    r = decide([row(k, 0.50, "rate")], [row(k, 0.499, "rate")], paired(0.50, 0.499, p=0.9, key=k))
    assert r["verdict"] == "PASS" and r["n_rows_paired"] == 1


def test_missing_row_fails():
    r = decide([row(), row("swap benchmark|de|far AUC", 0.99, "detection_auc")], [row()])
    assert r["verdict"] == "FAIL" and r["fails"][0]["rule"] == "missing row"


def test_lower_is_better_direction():
    k = "coaching tips on natives|es|tips per clip"
    b, c = row(k, 0.30, "other", hib=False), row(k, 0.20, "other", hib=False)
    r = decide([b], [c])
    assert r["verdict"] == "PASS" and not r["moved"][0]["worse"]


def test_deep_equal_ignores_timing_only():
    a = {"x": [1.0, 2.0], "timing_ms": {"total": 3.0}, "seconds": 0.2}
    b = {"x": [1.0, 2.0], "timing_ms": {"total": 9.0}, "seconds": 0.3}
    assert deep_equal(a, b) == []
    assert deep_equal(a, {**b, "x": [1.0, 2.0000001]})


@pytest.mark.skipif(not (TABLES / "swap_ours_vs_azure.json").exists(), reason="tables not present")
def test_rows_match_the_scoreboard_tally():
    from bench.opt.rows import extract, scoreboard_tally

    rows = extract(TABLES)
    az = [(r["area"], r["lang"], r["measure"], r["cat"]) for r in rows if r["tally"]]
    sb = [(t["area"], t["lang"], t["measure"], t["cat"]) for t in scoreboard_tally(TABLES)]
    assert az == sb and len(az) == 84
    assert len({r["key"] for r in rows}) == len(rows)


def test_suite_plan_never_contacts_rivals():
    from bench.opt.suite import STEPS, check_plan

    check_plan()
    names = " ".join(" ".join(s.args) for s in STEPS)
    assert "lingolix" not in names and "--steps azure" not in names
    assert "scoreboard_tally.json" in STEPS[-1].outputs
