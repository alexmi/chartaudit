# ChartAudit: project report

Meta Global AI Developer Hackathon, Oct 26 to Nov 4, 2026. Solo, Indie Developer track, theme Build for Your World.

## The problem

Traders mark up charts every day: support and resistance levels, supply and demand zones, position boxes with an entry, a stop and a target. These markups are how many traders think, and they pile up by the thousands as screenshots.

Nobody checks them. Looking back at a chart, the lines always seem to work, because you see the touches that held and forget the ones that broke. Testing a line honestly means replaying the chart by hand, bar by bar, from the date it was drawn. Almost nobody does it, so traders cannot tell whether their lines (or the lines of the analysts they follow) carry any edge at all.

The question ChartAudit answers: **after a line was drawn, did price actually respect it, more often than a line drawn at random would have?**

## What we built

A pipeline that goes from a screenshot to a scored audit, plus a demo page.

1. **Read.** Muse Spark 1.3 (Meta Model API) reads the screenshot into structured JSON: ticker, date, the price axis labels and scale, and every drawn level, zone and position box with its price. Prices printed on the chart are copied; lines without a label are estimated from the axis and flagged as estimates.
2. **Check.** Before anything is scored, an axis check verifies the reading: the axis labels must be self consistent, every drawn price must sit on the axis, and the real closing price on the chart date must fall inside it. A chart that fails is set aside with the reason, because a misread axis would misplace every line.
3. **Score, without hindsight.** Each drawing is scored only on the daily prices after the chart date, over 20 trading days. A level holds when price touches it (within 0.25 ATR) without closing through it; it breaks on a close through it. Position boxes win if the target comes before the stop.
4. **Compare.** On the same bars, ChartAudit builds baselines: swing pivot levels, random levels on the same side and distance, and random levels anywhere on the chart. Every hold rate carries a 95% Wilson interval.
5. **Show.** A Gradio page: upload a chart and see what was read drawn over the real prices, with the audit table. Three built-in synthetic examples work with no upload and no API key.

Also built: Muse Spark extracts specific calls and trading rules from spoken trading transcripts (17 testable rules from two transcripts), so spoken calls can later be joined to the charts they refer to. 155 tests.

## How well the model reads charts

20 synthetic charts with known answers, same prompt and scorer for every model:

| | Muse Spark 1.3 | Muse Spark Open (Baseten) | Claude Opus 5.5 | Claude Fable 5.1 |
|---|---|---|---|---|
| Printed line prices read exactly | 45 of 45 | 45 of 45 | 45 of 45 | 45 of 45 |
| Every axis label right | 7 of 20 | 7 of 20 | 11 of 20 | 6 of 20 |
| Log scale recognized | 0 of 5 | 0 of 5 | 5 of 5 | 0 of 5 |
| Worst price error | 73 ticks | 58 ticks | 0 ticks | 0 ticks |

What this taught us: printed numbers are read very reliably; geometry (log axes, unlabeled box edges) is not. One position box was read upside down, with stop and target swapped. That is why the axis check exists and why estimated prices are flagged. The open weights model matched the API model on this set, at about a minute per chart.

## Results on real charts

51 marked up charts from one trader's published analysis (Sep 8, 2026; images kept private).

* Muse Spark read **all 51**. 122 of 186 drawn prices had no label and were estimated from the axis.
* **43 charts** had drawings to score, and **all 43 passed the axis check**.

Levels, scored on the 20 trading days after each chart:

| | levels | reached | held | broke | hold rate | 95% interval |
|---|---|---|---|---|---|---|
| Drawn by the trader | 91 | 21 | 5 | 16 | 24% | 11% to 45% |
| Swing pivots | 83 | 34 | 13 | 21 | 38% | 24% to 55% |
| Random, matched side and distance | 455 | 113 | 32 | 81 | 28% | 21% to 37% |
| Random, anywhere | 455 | 182 | 48 | 134 | 26% | 21% to 33% |

* **No detectable advantage on this sample.** Drawn levels held 24% of the time when price reached them, against 26% to 38% for pivot and random levels. The intervals overlap.
* **Zones:** drawn zones held 9 of 18 times (50%), against 59% for random zones and 68% for zones at swing pivots.
* **Position boxes:** one was reached, and it lost.

## What counts as success

* **It works end to end.** Screenshot in, scored audit out, in about a minute per chart, on real charts the model had never seen.
* **It is honest about itself.** Reading errors are measured on charts with known answers, misreads are caught before scoring, and every number sits next to a random baseline with an interval.
* **It answered a real question.** For this chart set, the drawn lines did no better than random ones. That is not a failure of the tool; it is the answer hindsight never gives, and it took minutes instead of days of manual replay.

## Limits

* One chart set and small samples (21 drawn levels reached). It settles nothing about this trader or about chart lines in general.
* Log scale axes are not yet recognized by Muse Spark; those charts need a pixel based check.
* Daily bars only; intraday charts are not scored yet.

## Next

* Run the four more chart sets that are already read.
* Join spoken calls from transcripts to the charts they point at.
* Detect log scale axes from pixel positions.

## Reproduce

Code: https://github.com/alexmi/chartaudit (MIT). Setup, the audit command and the model comparison command are in the README. The demo page runs locally with `pip install -e ".[demo]"` and `python app.py`.
