# Quant Learn

A personal learning repo for quantitative finance. Built with Python, tushare, and akshare.

## Project Structure

```
quant_learn/
├── common/              # Shared utilities
│   ├── constant.py      # Asset pool config (funds & stocks)
│   ├── dataloader.py    # Load parquet data into DataFrames
│   └── draw.py          # Matplotlib Chinese font config
├── downloader/          # Data downloaders
│   ├── stock.py         # StockDownloader (tushare pro_bar)
│   └── fund.py          # FundDownloader (akshare ETF)
├── risk_analysis/       # Risk & return analysis
│   ├── analysis.py      # Entry point: run all analyses
│   ├── correlation_heatmap.py
│   ├── cumulative_returns.py
│   ├── sharpe_ratio.py
│   └── efficient_fontier.py
├── notes/               # Learning notes (markdown)
├── data/                # Downloaded & analysis output (gitignored)
└── .env                 # Tushare token (gitignored)
```

## What It Does

- **Download** daily stock/fund data from tushare and akshare
- **Correlation Heatmap** -- see which assets move together
- **Cumulative Returns** -- track how 1 yuan invested grows over time
- **Sharpe Ratio** -- find the best risk-adjusted returns
- **Efficient Frontier** -- find the optimal portfolio weights

## Setup

1. Create a `.env` file in the project root:
   ```
   TUSHARE_TOKEN=your_token_here
   ```

2. Install dependencies:
   ```bash
   conda activate myTools
   pip install -r requirements.txt
   ```

3. Download data:
   ```bash
   python downloader/stock.py
   python downloader/fund.py
   ```

4. Run analysis:
   ```bash
   python risk_analysis/analysis.py
   ```

   Output charts are saved to `data/analysis/risk_analysis/`.
