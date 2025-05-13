import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Load technical data

def load_technical_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Load sentiment data
def load_sentiment_data(file_path):
    df = pd.read_csv(file_path)
    df['created_time'] = pd.to_datetime(df['created_time'])
    return df

# Build technical features
def create_technical_features(df):
    df = df.copy()
    df['price_change']     = df['close'].pct_change()
    df['high_low_ratio']   = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['ma5']              = df['close'].rolling(5).mean()
    df['ma10']             = df['close'].rolling(10).mean()
    df['ma20']             = df['close'].rolling(20).mean()
    df['volatility']       = df['close'].rolling(20).std()
    df['volume_change']    = df['volume'].pct_change()
    df['returns']          = df['close'].pct_change()
    df['log_returns']      = np.log(df['close'] / df['close'].shift())
    return df

# Merge technical and sentiment data, only keep diagnostic period 2023-01-01 to 2024-06-30
def merge_data(tech_df, sent_df):
    start, end = '2023-01-01', '2024-06-30'
    tech = tech_df[(tech_df['timestamp'] >= start) & (tech_df['timestamp'] <= end)]
    sent = sent_df[(sent_df['created_time'] >= start) & (sent_df['created_time'] <= end)]
    merged = pd.merge_asof(
        tech.sort_values('timestamp'),
        sent.sort_values('created_time'),
        left_on='timestamp',
        right_on='created_time',
        direction='nearest'
    )
    return merged

# Calculate advanced factors, and add dual_momentum²
def calculate_advanced_factors(df):
    df = df.copy()
    N = 20
    k = 5
    # Sentiment shock
    df['sentiment_mean'] = df['sentiment_score'].rolling(N).mean()
    df['sentiment_std']  = df['sentiment_score'].rolling(N).std()
    df['sentiment_shock'] = (df['sentiment_score'] - df['sentiment_mean']) / df['sentiment_std']
    # Sentiment and volume resonance
    df['volume_ma'] = df['volume'].rolling(N).mean()
    df['emo_vol_resonance'] = df['sentiment_score'] * (df['volume_ma'] / df['volume'])
    # Awareness factor
    df['sentiment_snr']      = df['sentiment_score'].abs() / df['sentiment_std']
    df['weighted_sentiment'] = df['sentiment_score'] * (df['volume'] / df['volume_ma'])
    # Dual momentum
    df['price_mom']      = df['close'].pct_change(k)
    df['sent_mom']       = df['sentiment_score'].diff(k)
    df['dual_momentum']  = df['price_mom'] * df['sent_mom']
    df['dual_momentum_sq'] = df['dual_momentum'] ** 2  # Add quadratic term feature
    df['momentum_diff']  = df['sent_mom'] - df['price_mom']
    # Extreme resonance
    delta = df['close'].diff()
    gain  = delta.where(delta>0, 0).rolling(14).mean()
    loss  = -delta.where(delta<0, 0).rolling(14).mean()
    rs    = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)
    low_q  = df['sentiment_score'].quantile(0.1)
    high_q = df['sentiment_score'].quantile(0.9)
    df['extreme_resonance'] = (
        ((df['sentiment_score'] <= low_q) & (df['rsi'] < 30)) |
        ((df['sentiment_score'] >= high_q) & (df['rsi'] > 70))
    )
    df['extreme_resonance_strength'] = 0.0
    mask = df['extreme_resonance']
    df.loc[mask, 'extreme_resonance_strength'] = (
        ((df.loc[mask,'sentiment_score'] - high_q) /
         (df['sentiment_score'].quantile(0.99) - high_q)).fillna(0)
        + ((df.loc[mask,'rsi'] - 70) / 30).fillna(0)
    )
    # Rhythm sync
    df['volume_change']    = df['volume'].pct_change()
    df['sentiment_change'] = df['sentiment_score'].diff()
    df['rhythm_sync']      = df['volume_change'].rolling(N).corr(df['sentiment_change'])
    # PCA factor
    features = [
        'price_mom','sent_mom','dual_momentum','dual_momentum_sq','momentum_diff',
        'sentiment_shock','emo_vol_resonance','sentiment_snr','weighted_sentiment',
        'extreme_resonance_strength','rhythm_sync','volatility'
    ]
    X = df[features].dropna()
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)
    df.loc[X.index, 'pca1'] = comp[:,0]
    df.loc[X.index, 'pca2'] = comp[:,1]
    # Trend confirmation
    df['tech_signal'] = np.where(df['rsi'] > 70, 1, np.where(df['rsi'] < 30, -1, 0))
    df['sent_signal'] = np.where(df['sentiment_score'] > df['sentiment_score'].quantile(0.8), 1,
                          np.where(df['sentiment_score'] < df['sentiment_score'].quantile(0.2), -1, 0))
    df['signal_strength_ratio'] = df['price_mom'].diff() / df['sentiment_score'].diff()
    return df

# Main process, iterate through each period and 1-12x prediction horizon

def main():
    tech_files = {
        '1h': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1h-futures.json',
        '4h': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-4h-futures.json',
        '1d': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.json',
        '1w': 'ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1w-futures.json'
    }
    sent_files = {
        '1h': 'ft_userdata/user_data/data/processed/sentiment_features_1H.csv',
        '4h': 'ft_userdata/user_data/data/processed/sentiment_features_4H.csv',
        '1d': 'ft_userdata/user_data/data/processed/sentiment_features_1D.csv',
        '1w': 'ft_userdata/user_data/data/processed/sentiment_features_1W.csv'
    }
    period_hours = {'1h':1, '4h':4, '1d':24, '1w':168}

    for feat_period, tech_path in tech_files.items():
        tech = load_technical_data(tech_path)
        sent = load_sentiment_data(sent_files[feat_period])
        tech = create_technical_features(tech)
        merged = merge_data(tech, sent)
        merged = calculate_advanced_factors(merged)

        feat_h = period_hours[feat_period]
        print(f"=== Feature period: {feat_period} ===\n")
        for step in range(1, 8):
            fut_h = feat_h * step
            # future returns
            merged[f'future_{fut_h}h_returns'] = merged['close'].pct_change(periods=step).shift(-step)
            cols = [
                'sentiment_shock', 'emo_vol_resonance', 'sentiment_snr', 'weighted_sentiment',
                'dual_momentum', 'dual_momentum_sq', 'momentum_diff', 'extreme_resonance_strength',
                'rhythm_sync', 'pca1', 'pca2', 'signal_strength_ratio'
            ]
            sub = merged[cols + [f'future_{fut_h}h_returns']].dropna()
            lin_corr  = sub[cols].corrwith(sub[f'future_{fut_h}h_returns']).sort_values(ascending=False)
            quad_corr = sub[cols].pow(2).corrwith(sub[f'future_{fut_h}h_returns']).sort_values(ascending=False)

            print(f"-- Predicted {fut_h} hours later returns ({feat_period} x {step}) --")
            print("Linear correlation:")
            print(lin_corr)
            print("Quadratic correlation:")
            print(quad_corr)
            print()

if __name__ == '__main__':
    main()
