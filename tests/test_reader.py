from types import SimpleNamespace

import pytest

from chartaudit.reader import ReadError, make_client, parse_reading, read_chart

GOOD = (
    '{"ticker":"META","timeframe":"4h","last_visible_date":"2025-05-16","y_axis_scale":"linear",'
    '"y_axis_ticks":["640","620"],"drawings":[{"type":"position_box","entry":621.11,"stop":612.2,"target":735.1}]}'
)


def test_parse_plain_json():
    r = parse_reading(GOOD)
    assert r.ticker == "META" and r.drawings[0].entry == 621.11


def test_parse_fenced_json_with_prose():
    assert parse_reading("Here you go:\n```json\n" + GOOD + "\n```").timeframe == "4h"


def test_parse_nulls_allowed():
    r = parse_reading(
        '{"ticker":null,"timeframe":null,"last_visible_date":null,"y_axis_scale":null,"y_axis_ticks":[],'
        '"drawings":[{"type":"position_box","entry":10,"stop":null,"target":null}]}'
    )
    assert r.ticker is None and r.drawings[0].stop is None


def test_parse_garbage_raises():
    with pytest.raises(ReadError):
        parse_reading("I cannot read this chart.")


def test_parse_wrong_shape_raises():
    with pytest.raises(ReadError):
        parse_reading('{"drawings":"a box at 100"}')
    assert parse_reading('{"drawings":[{"type":"banana"}]}').drawings[0].type == "other"


def test_read_chart_sends_image(tmp_path):
    img = tmp_path / "c.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    sent = {}

    def create(**kw):
        sent.update(kw)
        msg = SimpleNamespace(content=GOOD)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reading, raw = read_chart(img, fake, "muse-spark-1.3")
    assert reading.ticker == "META" and raw == GOOD
    assert sent["model"] == "muse-spark-1.3" and sent["temperature"] == 0
    assert sent["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_make_client_needs_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", "")
    with pytest.raises(ReadError, match="MODEL_API_KEY not set"):
        make_client()


def test_lenient_formats_are_accepted():
    r = parse_reading(
        '{"ticker":"NQ1!","timeframe":240,"last_visible_date":null,"y_axis_scale":"Linear",'
        '"y_axis_ticks":[640,620.5],"drawings":[{"type":"horizontal_line","price":"29,868.00"},'
        '{"type":"rectangle","top":10,"bottom":9}]}'
    )
    assert r.timeframe == "240" and r.y_axis_scale == "linear"
    assert r.y_axis_ticks == ["640", "620.5"]
    assert r.drawings[0].price == 29868.0 and r.drawings[1].type == "other"


def test_null_lists_become_empty():
    r = parse_reading('{"ticker":"X","y_axis_ticks":null,"drawings":null}')
    assert r.y_axis_ticks == [] and r.drawings == []


def test_read_error_keeps_raw_text(tmp_path):
    img = tmp_path / "c.png"
    img.write_bytes(b"x")
    bad = "Sorry, I can't help with that."
    create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=bad))])  # noqa: E731
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ReadError) as e:
        read_chart(img, fake, "m")
    assert e.value.raw == bad


def test_client_has_timeout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", "test-not-a-real-key")
    client, _ = make_client()
    assert client.timeout == 60


def test_baseten_provider_uses_its_own_key_url_and_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    client, model = make_client("baseten")
    assert str(client.base_url).startswith("https://inference.baseten.co/v1")
    assert model == "meta-models/muse-spark-open-1026"


def test_baseten_provider_needs_its_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("BASETEN_API_KEY", "")
    with pytest.raises(ReadError, match="BASETEN_API_KEY"):
        make_client("baseten")


def test_unknown_provider_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ReadError, match="unknown provider"):
        make_client("nope")


@pytest.mark.parametrize("provider,key_name,base,model", [
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-5.5"),
    ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-opus-5-5"),
])
def test_other_providers_use_their_own_key_and_url(monkeypatch, tmp_path, provider, key_name, base, model):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(key_name, "test-not-a-real-key")
    client, got = make_client(provider)
    assert str(client.base_url).startswith(base) and got == model


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    _, got = make_client("baseten", model="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B")
    assert got == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
