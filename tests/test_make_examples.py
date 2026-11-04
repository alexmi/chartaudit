from chartaudit.examples import list_examples, run_example
from make_examples import EXAMPLES, build, find_seed, trader_outcomes


def test_each_example_kind_finds_a_seed_with_that_outcome():
    for ex in EXAMPLES:
        outcomes = trader_outcomes(ex, find_seed(ex))
        assert outcomes and all(o == ex["want"] for o in outcomes), (ex["name"], outcomes)


def test_build_saves_runnable_examples_from_the_reader(tmp_path):
    seen = []
    build(tmp_path, read=lambda png, truth: seen.append(png.name) or truth)
    assert list_examples(tmp_path) == sorted(e["name"] for e in EXAMPLES)
    assert len(seen) == len(EXAMPLES)
    for e in EXAMPLES:
        _, table, _, _ = run_example(e["name"], tmp_path)
        assert table and table[0]["source"] == "trader"


def test_examples_keep_the_chart_date_when_the_model_reads_none(tmp_path):
    build(tmp_path, read=lambda png, truth: truth.model_copy(update={"last_visible_date": None}))
    for e in EXAMPLES:
        _, table, _, _ = run_example(e["name"], tmp_path)
        assert table, e["name"]
