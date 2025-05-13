# Crypto Quant Strategy Backtesting (with Freqtrade)

This repo contains multiple quantitative trading strategies for the crypto market, including LSTM-based models that integrate technical and sentiment features.

## Quick Start (Test Mode)

If you only want to test the strategy logic:

### What You Need

Download these files only:

- `ft_userdata/user_data/models/` — pre-trained model files
- `ft_userdata/user_data/strategies/` — strategy Python scripts
- `ft_userdata/user_data/data/` — all processed input data (including sentiment features and price data)
- `config.json` — your Freqtrade config

### Set Up Docker (with Freqtrade)

1. Open Docker.
2. Search for `freqtrade` in Docker Hub.
3. Download the image with tag: `stable_freqaitorch`
4. Follow the [official quickstart guide](https://www.freqtrade.io/en/stable/docker_quickstart/) to install Freqtrade inside the container.

```markdown
### Run Backtesting

Once inside the Freqtrade container, run the following commands:

```bash
freqtrade backtesting --strategy LstmStrategy \
    --timerange 20240701-20250331 \
    --datadir user_data/data/binance \
    --timeframe 1h -v

freqtrade backtesting --strategy LstmWithSentimentStrategy \
    --timerange 20240701-20250331 \
    --datadir user_data/data/binance \
    --timeframe 1d -v


Note:

LstmWithSentimentStrategy uses a 1d timeframe (because of how the sentiment data is saved).

LstmStrategy uses 1h since technical-only strategies are more flexible on timeframes.

Some script files might contain absolute paths specific to my machine (e.g., /Users/...). You can edit them to match your environment if needed.



## Full Workflow (Reproducible Pipeline)

To fully replicate the full pipeline from raw data to model training and strategy execution:

1. Scrape Forum Data
Use the .py files inside the two crawler folders.

Running them will automatically crawl forum posts.

These scraped files are not uploaded due to size.

Place the scraped .csv or .json files under:
ft_userdata/user_data/data/topic/

2. Generate Sentiment Features
Run:
python ft_userdata/user_data/scripts/process_sentiment_features.py
This will produce a structured dataset in:
ft_userdata/user_data/data/

3. Train Your Own Models
Run one of the training scripts depending on your desired model:
python ft_userdata/user_data/freqaimodels/lstm_training.py

# or
python ft_userdata/user_data/freqaimodels/lstm_training_with_sentiment.py
After training, the following directory will contain your models:

ft_userdata/user_data/models/
