import pytest

from chartaudit.calls import CallSheet, parse_calls


def test_calls_and_rules_parse_from_model_json():
    text = """Here you go: {"date": "2026-10-28", "calls": [
      {"ticker": "spy", "action": "buy", "kind": "zone", "top": "770.5", "bottom": 765, "timeframe": "60m",
       "condition": "on a pullback", "quote": "There's now a new demand zone below."},
      {"ticker": "GOOGL", "action": "watch", "kind": "zone", "top": null, "bottom": null, "quote": "demand zone from April"}],
     "rules": [{"rule": "First test of supply is sold", "quote": "first test of supply is always sold"}]}"""
    sheet = parse_calls(text)
    assert isinstance(sheet, CallSheet) and sheet.date == "2026-10-28"
    spy = sheet.calls[0]
    assert spy.ticker == "SPY" and spy.action == "buy" and (spy.top, spy.bottom) == (770.5, 765.0)
    assert sheet.calls[1].top is None
    assert sheet.rules[0].rule.startswith("First test")


def test_unknown_action_becomes_other():
    sheet = parse_calls('{"calls": [{"ticker": "X", "action": "scalp it"}], "rules": []}')
    assert sheet.calls[0].action == "other"


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_calls("sorry, no calls")
