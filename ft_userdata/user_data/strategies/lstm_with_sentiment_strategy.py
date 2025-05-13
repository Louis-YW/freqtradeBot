"""
Freqtrade futures strategy driven by an LSTM price‑prediction model
(Technical + sentiment features), combined with MACD / OBV / ATR conditions.

Usage
-----
1. Run the *trainer* script to train the model, which will generate
   `price_lstm_with_sentiment.pt` and corresponding scaler
   (`feature_scaler.joblib` / `target_scaler.joblib`).
2. Put this file into `user_data/strategies/` and specify it in `config.json`:
       "strategy": "LstmWithSentimentStrategy"
3. Backtest example:
       freqtrade backtesting \
           --strategy LstmWithSentimentStrategy \
           --timeframe 1h \
           --timerange 20240701-20250331 \
           --datadir user_data/data/binance \
           --break-cache -v

Attention
----
* The model input window is 24 K lines, with 30 features (including 12 sentiment features).
* Use a safe model loading method.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
#  LSTM structure —— same as the training script
# ---------------------------------------------------------------------------
class PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Modify the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Get the output of the last time step
        last_hidden = out[:, -1]
        
        # Apply batch normalization
        last_hidden = self.bn1(last_hidden)
        
        # Fully connected layer
        return self.fc(last_hidden)


# ---------------------------------------------------------------------------
#  Simplified technical indicators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    diff = series.diff()
    up = diff.clip(lower=0).rolling(period).mean()
    down = -diff.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    m = ema_fast - ema_slow
    return m, m.ewm(span=signal, adjust=False).mean()


def bollinger(series: pd.Series, window: int = 20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma
    return upper, lower, width


def atr(df: DataFrame, period: int = 14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(df: DataFrame):
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volatility(series: pd.Series, window: int = 24):
    return series.pct_change().rolling(window).std()


# ---------------------------------------------------------------------------
#  Strategy
# ---------------------------------------------------------------------------
class LstmWithSentimentStrategy(IStrategy):
    """Freqtrade strategy – LSTM price prediction + sentiment + technical filtering"""

    # === Basic settings ===
    timeframe = "1d"
    can_short = True
    startup_candle_count = 30

    trading_mode = "futures"
    margin_mode = "isolated"
    default_leverage = 1.0

    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    process_only_new_candles = True

    # === Fixed position settings ===
    stake_amount = 100            # Fixed 100 USDT per trade
    position_staking = False      # Do not use position staking
    max_open_trades = 3          # Maximum number of open trades at the same time

    # === Trading settings ===
    use_exit_signal = True           # Use exit signal
    exit_profit_only = False         # Only exit when profitable
    ignore_roi_if_entry_signal = True # Ignore ROI if there is an entry signal
    position_adjustment_enable = False # Disable position adjustment

    long_threshold = 0.005   # +0.5 %
    short_threshold = -0.005 # -0.5 %
    WINDOW_SIZE = 10          # Same as training

    # === Model files ===
    _model_dir = Path("user_data/models/lstm_with_sentiment")
    model_path = _model_dir / "price_lstm_with_sentiment.pt"
    feat_scaler_path = _model_dir / "feature_scaler.joblib"
    targ_scaler_path = _model_dir / "target_scaler.joblib"

    # === Feature columns ===
    sentiment_cols = [
        "emo_vol_resonance", "signal_strength_ratio", "pca1", "rhythm_sync",
        "dual_momentum_sq", "dual_momentum", "momentum_diff", "sentiment_shock",
        "pca2", "sentiment_snr", "weighted_sentiment", "extreme_resonance_strength",
    ]

    feature_cols: List[str] = [
        "close", "open", "high", "low", "volume",
        "return_1", "return_3", "return_6", "rsi", "macd",
        "macd_signal", "bb_upper", "bb_lower", "bb_width",
        "atr", "volatility_24", "obv", "mom_10",
    ] + sentiment_cols

    # ---------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Scaler
        self.x_scaler = joblib.load(self.feat_scaler_path)
        self.y_scaler = joblib.load(self.targ_scaler_path)

        # 2) Model – Load
        try:
            # Load model checkpoint
            chkpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model instance
            self.model = PriceLSTM(len(self.feature_cols)).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(chkpt['model_state_dict'])
            self.model.eval()
            
            # Print model information
            print("\n=== Model information ===")
            print(f"Model device: {self.device}")
            print(f"Model training mode: {self.model.training}")
            print(f"Model parameter count: {sum(p.numel() for p in self.model.parameters())}")
            print("================\n")
            
        except Exception as e:
            print(f"Error: Failed to load model: {e}")
            raise

        # 3) Sentiment CSV (optional)
        csv_path = Path("user_data/data/processed/sentiment_features_1D.csv")
        if csv_path.exists():
            df_sent = pd.read_csv(csv_path)
            df_sent["date"] = pd.to_datetime(df_sent["created_time"])
            self.sentiment_df = (
                df_sent.set_index("date")
                .sort_index()
                .ffill()
                .bfill()
            )
        else:
            self.sentiment_df = pd.DataFrame()

    # ---------------------------------------------------------------------
    #  Indicator calculation
    # ---------------------------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # --- Technical factors ---
        df["return_1"] = df["close"].pct_change()
        df["return_3"] = df["close"].pct_change(3)
        df["return_6"] = df["close"].pct_change(6)
        df["rsi"] = rsi(df["close"])
        df["macd"], df["macd_signal"] = macd(df["close"])
        df["bb_upper"], df["bb_lower"], df["bb_width"] = bollinger(df["close"])
        df["atr"] = atr(df)
        df["volatility_24"] = volatility(df["close"])
        df["obv"] = obv(df)
        df["mom_10"] = df["close"] - df["close"].shift(10)

        # --- Initialize sentiment columns to 0 ---
        for col in self.sentiment_cols:
            df[col] = 0.0

        # --- If there is sentiment data, generate sentiment features ---
        if not self.sentiment_df.empty:
            # Ensure both dataframes have datetime index
            df.index = pd.to_datetime(df.index)
            self.sentiment_df.index = pd.to_datetime(self.sentiment_df.index)
            
            # Merge data
            df = pd.merge_asof(
                df.sort_index(),
                self.sentiment_df[["sentiment_score"]].sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
                tolerance=pd.Timedelta("1D"),
            )

            s = df["sentiment_score"]
            if s.notna().any():
                N, k = 20, 5
                df["sentiment_mean"] = s.rolling(N).mean()
                df["sentiment_std"] = s.rolling(N).std()
                df["sentiment_shock"] = (s - df["sentiment_mean"]) / df["sentiment_std"]

                df["volume_ma"] = df["volume"].rolling(N).mean()
                df["emo_vol_resonance"] = s * (df["volume_ma"] / df["volume"])

                df["sentiment_snr"] = s.abs() / df["sentiment_std"]
                df["weighted_sentiment"] = s * (df["volume"] / df["volume_ma"])

                df["price_mom"] = df["close"].pct_change(k)
                df["sent_mom"] = s.diff(k)
                df["dual_momentum"] = df["price_mom"] * df["sent_mom"]
                df["dual_momentum_sq"] = df["dual_momentum"].pow(2)
                df["momentum_diff"] = df["sent_mom"] - df["price_mom"]

                low_q, high_q = s.quantile(0.1), s.quantile(0.9)
                df["extreme_resonance"] = (
                    ((s <= low_q) & (df["rsi"] < 30)) |
                    ((s >= high_q) & (df["rsi"] > 70))
                )
                df["extreme_resonance_strength"] = 0.0
                m = df["extreme_resonance"]
                if m.any():
                    sent_strength = ((s - high_q) / (s.quantile(0.99) - high_q)).fillna(0)
                    rsi_strength = ((df["rsi"] - 70) / 30).fillna(0)
                    df.loc[m, "extreme_resonance_strength"] = sent_strength + rsi_strength

                df["volume_change"] = df["volume"].pct_change()
                df["sentiment_change"] = s.diff()
                df["rhythm_sync"] = df["volume_change"].rolling(N).corr(df["sentiment_change"])

                pca_feat = [
                    "price_mom", "sent_mom", "dual_momentum", "dual_momentum_sq",
                    "momentum_diff", "sentiment_shock", "emo_vol_resonance",
                    "sentiment_snr", "weighted_sentiment", "extreme_resonance_strength",
                    "rhythm_sync",
                ]
                X = df[pca_feat].dropna()
                if not X.empty:
                    comp = PCA(n_components=2).fit_transform(X)
                    df.loc[X.index, "pca1"] = comp[:, 0]
                    df.loc[X.index, "pca2"] = comp[:, 1]

                df["signal_strength_ratio"] = df["price_mom"].diff() / s.diff()

        # Clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0.0, inplace=True)

        return df

    # ------------------------------------------------------------------
    #  Internal prediction
    # ------------------------------------------------------------------
    def _predict_price(self, df: DataFrame) -> float:
        if len(df) < self.WINDOW_SIZE:
            return np.nan
        window = df[self.feature_cols].iloc[-self.WINDOW_SIZE:].values.astype(np.float32)
        scaled = self.x_scaler.transform(window).reshape(1, self.WINDOW_SIZE, -1)
        with torch.no_grad():
            pred_norm = self.model(torch.tensor(scaled, device=self.device)).item()
        return float(self.y_scaler.inverse_transform([[pred_norm]])[0, 0])

    # ------------------------------------------------------------------
    #  ENTRY
    # ------------------------------------------------------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_short"] = 0

        pred = self._predict_price(df)
        df.at[df.index[-1], "pred_price"] = pred
        if np.isnan(pred):
            return df

        price = df["close"].iloc[-1]
        delta = (pred - price) / price

        # Print debug information
        print(f"\n=== Trading signal check ===")
        print(f"Current time: {df.index[-1]}")
        print(f"Current price: {price:.2f}")
        print(f"Predicted price: {pred:.2f}")
        print(f"Price deviation: {delta:.4%}")
        print(f"MACD: {df['macd'].iloc[-1]:.2f}")
        print(f"MACD signal: {df['macd_signal'].iloc[-1]:.2f}")
        print(f"OBV current: {df['obv'].iloc[-1]:.2f}")
        print(f"OBV previous: {df['obv'].shift(8).iloc[-1]:.2f}")
        print(f"Sentiment score: {df.get('sentiment_score', pd.Series(0, index=df.index)).iloc[-1]:.2f}")

        # Long conditions
        long_conditions = (
            (delta > self.long_threshold) &  # Predicted price上涨
            (df['macd'] > df['macd_signal']) &  # MACD golden cross
            (df['macd'] > 0) &  # MACD above zero axis
            (df['obv'] > df['obv'].shift(8)) &  # OBV upward trend
            (df['obv'] > df['obv'].shift(14)) &  # OBV medium-term upward trend
            (df['close'] > df['close'].shift(7))  # Price 7-period upward trend
        )

        # Short conditions
        short_conditions = (
            (delta < self.short_threshold) &  # Predicted price下跌
            (df['macd'] < df['macd_signal']) &  # MACD death cross
            (df['macd'] < 0) &  # MACD below zero axis
            (df['obv'] < df['obv'].shift(8)) &  # OBV downward trend
            (df['obv'] < df['obv'].shift(14)) &  # OBV medium-term downward trend
            (df['close'] < df['close'].shift(7))  # Price 7-period downward trend
        )

        # Set entry signals
        df.loc[long_conditions, 'enter_long'] = 1
        df.loc[short_conditions, 'enter_short'] = 1

        # Print signal status
        print(f"Long signal: {df.loc[df.index[-1], 'enter_long']}")
        print(f"Short signal: {df.loc[df.index[-1], 'enter_short']}")
        return df

    # ------------------------------------------------------------------
    #  EXIT
    # ------------------------------------------------------------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_short"] = 0

        pred = df.get("pred_price", pd.Series(np.nan, index=df.index)).iloc[-1]
        if np.isnan(pred):
            return df

        price = df["close"].iloc[-1]
        delta = (pred - price) / price
        macd_val = df["macd"].iloc[-1]
        macd_sig = df["macd_signal"].iloc[-1]

        if delta < 0 or macd_val < macd_sig:
            df.at[df.index[-1], "exit_long"] = 1
        if delta > 0 or macd_val > macd_sig:
            df.at[df.index[-1], "exit_short"] = 1
        return df

    # ------------------------------------------------------------------
    #  Leverage (fixed 1x, adjustable as needed)
    # ------------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return self.default_leverage
