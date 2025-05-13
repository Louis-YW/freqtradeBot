import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Load technical data
def load_technical_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Build technical features (use min_periods=1 throughout to avoid long-period NaNs)
def create_technical_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Basic price/volume changes
    df['price_change']     = df['close'].pct_change().fillna(0)
    df['volume_change']    = df['volume'].pct_change().fillna(0)
    df['high_low_ratio']   = (df['high'] / df['low']).fillna(1)
    df['close_open_ratio'] = (df['close'] / df['open']).fillna(1)

    # Moving averages
    for n in [5, 10, 20, 50, 100]:
        df[f'ma{n}'] = df['close'].rolling(n, min_periods=1).mean()

    # Volatility (20)
    df['volatility20'] = df['close'].rolling(20, min_periods=1).std().fillna(0)

    # ATR14
    high_low = df['high'] - df['low']
    high_cp  = (df['high'] - df['close'].shift()).abs()
    low_cp   = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14, min_periods=1).mean().fillna(0)

    # RSI14
    delta = df['close'].diff().fillna(0)
    gain  = delta.where(delta>0, 0).rolling(14, min_periods=1).mean()
    loss  = -delta.where(delta<0, 0).rolling(14, min_periods=1).mean()
    rs    = gain.div(loss.replace(0, np.nan)).fillna(0)
    df['rsi14'] = 100 - 100 / (1 + rs)

    # MACD (12,26,9)
    ema12 = df['close'].ewm(span=12, min_periods=1).mean()
    ema26 = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd_line']   = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, min_periods=1).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']

    # Bollinger Bands (20,2σ)
    ma20 = df['close'].rolling(20, min_periods=1).mean()
    std20 = df['close'].rolling(20, min_periods=1).std().fillna(0)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df['bollinger_bw'] = ((upper - lower) / ma20).fillna(0)

    # OBV
    df['obv'] = (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum().fillna(0)

    return df

# Main process: iterate through each period and 1-12x prediction horizon
def main():
    tech_files = {
        '1h': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1h-futures.json',
        '4h': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-4h-futures.json',
        '1d': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.json',
        '1w': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1w-futures.json'
    }
    period_hours = {'1h':1, '4h':4, '1d':24, '1w':168}

    # All technical features
    tech_feats = [
        'price_change','volume_change','high_low_ratio','close_open_ratio',
        'ma5','ma10','ma20','ma50','ma100',
        'volatility20','atr14','rsi14',
        'macd_line','macd_signal','macd_hist',
        'bollinger_bw','obv'
    ]

    for feat_period, tech_path in tech_files.items():
        df = load_technical_data(tech_path)
        df = create_technical_features(df)

        feat_h = period_hours[feat_period]
        print(f"\n=== Feature period: {feat_period} ===\n")

        for step in range(1, 8):
            fut_h = feat_h * step
            # future returns
            df[f'future_{fut_h}h_returns'] = df['close'].pct_change(periods=step).shift(-step)

            sub = df[tech_feats + [f'future_{fut_h}h_returns']].dropna()
            if sub.empty:
                print(f"-- Predicted {fut_h} hours later returns ({feat_period} x {step}) -- All NaN, skip")
                continue

            lin_corr  = sub[tech_feats].corrwith(sub[f'future_{fut_h}h_returns']).sort_values(ascending=False)
            quad_corr = sub[tech_feats].pow(2).corrwith(sub[f'future_{fut_h}h_returns']).sort_values(ascending=False)

            print(f"-- Predicted {fut_h} hours later returns ({feat_period} x {step}) --")
            print("Linear correlation:")
            print(lin_corr.round(4))
            print("Quadratic correlation:")
            print(quad_corr.round(4))
            print()

if __name__ == '__main__':
    main()
