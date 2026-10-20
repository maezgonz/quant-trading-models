# Backtest-to-Live Checklist

> Adapted from Vidovich & Zerjal (arXiv:2610.21846), *"Where Backtests and Live Trading Part Ways: Six Defects from One Intraday System."* Each item guards against a defect they found in production. This checklist is applied to every backtest in this repository before results are published.

## The 10-point checklist

### Data integrity

| # | Check | What went wrong in the paper | How we guard against it |
|---|---|---|---|
| 1 | **Volume fidelity**: does the backtest use the same volume data the live system sees? | Default feed carried only a fraction of real opening volume | Our data layer (`src/data_ingestion.py`) uses a single source for both backtest and report; volume fields are validated for > 0 and against known ranges |
| 2 | **Price adjustment consistency**: are historical prices adjusted the same way the live system prices? | Backtest used dividend-adjusted prices; live orders saw raw prices | We document which price field is used (`Close` vs `Adj Close`) in every module's docstring; no implicit adjustment |
| 3 | **Data completeness**: is the backtest consuming all data it downloads? | Live system downloaded data the backtest never used | Every `fetch_*` call feeds directly into the computation; no orphan data paths |

### Code correctness

| # | Check | What went wrong in the paper | How we guard against it |
|---|---|---|---|
| 4 | **Position sizing limits**: does the backtest enforce the same limits as the live account? | Sizing step exceeded a limit the live account enforced | `strategies/sma_crossover.py` caps exposure at 100% (long/flat); no leverage assumed |
| 5 | **Timezone discipline**: are timestamps interpreted consistently? | UTC timestamps were read as New York time | All timestamps are timezone-naive daily dates; no intraday timezones in this repository |
| 6 | **Lookahead bias**: does any computation use information not available at decision time? | (Classical defect, implied) | `sma_crossover.py` explicitly shifts signals by 1 day (`position = signal.shift(1)`) — positions are taken the day after the signal fires |

### Configuration alignment

| # | Check | What went wrong in the paper | How we guard against it |
|---|---|---|---|
| 7 | **Config propagation**: do settings actually reach the code that uses them? | Settings were switched on but never reached order-placing code | CLI arguments are passed directly to the computation functions; no external config files that could silently diverge |
| 8 | **Backtest-live replay parity**: can you replay the backtest with live data and get the same decisions? | (The core methodology of the paper) | Each backtest module is deterministic given (data, seed); seed is configurable and logged |

### Statistical honesty

| # | Check | What went wrong in the paper | How we guard against it |
|---|---|---|---|
| 9 | **Execution realism**: does the backtest assume fills the live system wouldn't get? | (Related to adverse selection) | Our SMA crossover backtest uses close-to-close returns (no intraday fill assumptions); for market-order strategies, we note that slippage is not modeled |
| 10 | **Summary statistics can hide defects**: do you inspect per-day/per-trade results, not just aggregate metrics? | "The drift never showed in summary statistics" | Our tear sheets (`src/tear_sheet.py`) include daily return distributions, drawdown timelines, and monthly heatmaps — not just aggregate Sharpe/return |

## How to use this checklist

Before publishing any backtest result in this repository:

1. Run through each item and note ✅/❌/N/A
2. If any item is ❌, fix it before committing
3. Document any N/A items with a brief justification in the README section of the strategy
