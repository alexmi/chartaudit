# ChartAudit: eval ideas

## 1. Random-walk null markets (from Martingale Doppelganger-Eval, saved 2026-10-27)
**Source:** arXiv 2606.17423 (q-fin.CP, Jun 16, 2026): https://arxiv.org/abs/2606.17423

**What they do:** synthetic "shadow markets" where price is a random walk (a martingale: no real edge exists), plus paired versions with a small edge injected. 60-candle inputs, 5-step horizon. Tests whether a vision model actually reads the candlestick evidence or just extrapolates the visible trend.

**How ChartAudit uses it:**
* **Null test for the scorer:** draw lines on random-walk charts. Any "edge" the scorecard reports there is a bug or luck; this is the false-positive rate.
* **Injected-edge test:** add a known edge (e.g. price bounces at a planted level with probability p) and check the scorecard finds it, and how many charts that takes. This is the power, and tells us when to say "can't tell yet".
* **Baselines:** trader lines vs algorithmic levels vs random levels, all on the same charts.
* **Demo line:** "On fake markets with no edge, ChartAudit says no edge. On real charts, here's what it found."

## 2. Balance winners and losers on purpose
Public posts skew toward winners: TradingView authors mark "target reached" far more often than "stop reached". Sample good and bad outcomes deliberately, and score everything from price history after the post date, never from the author's status. Same hindsight bias ChartAudit exists to catch.

## 3. Rights
* Private eval (`data/private/`, gitignored): any image, including copyrighted ones, kept local.
* Public repo: public domain and CC images only (credited; CC BY-SA kept in its own folder because share-alike carries over), plus synthetic charts we render ourselves. Copyrighted examples appear only as links and labels.
