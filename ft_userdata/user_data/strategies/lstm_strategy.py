"""
Freqtrade futures strategy driven by an LSTM price-prediction model
*plus* MACD / OBV / ATR confirmations.

Prerequisites
-------------
1. Train the model and saves *.pt / *.joblib under user_data/models/lstm/
2. Copy this file to user_data/strategies/lstm_strategy.py
3. In your config.json:
       "strategy": "LstmStrategy",
       "timeframe": "1h",
       "exchange": { "name": "binance", "options": { "defaultType": "swap" } }

Notes
-----
* Model input shape = (1, 24, 18)  → last 24 candles, 18 features
* `_predict()` returns the **predicted absolute price**.
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.serialization
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from typing import List
from datetime import datetime

# Add security loading settings
torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])

# -----------------------------------------------------------------------------
# Model (same architecture as the trainer)
# -----------------------------------------------------------------------------
class PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
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

# -----------------------------------------------------------------------------
# Technical indicators (identical to trainer)
# -----------------------------------------------------------------------------
import pandas as pd

def calculate_rsi(prices: pd.Series, period: int = 14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger(prices: pd.Series, window: int = 20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma
    return upper, lower, width

def calculate_atr(df: DataFrame, period: int = 14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_obv(df: DataFrame):
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()

def calculate_volatility(prices: pd.Series, window: int = 24):
    return prices.pct_change().rolling(window).std()

# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------
class LstmStrategy(IStrategy):
    """
    Long / short entries when:
        predicted_price deviation ≥ ±0.1 %
        + MACD confirmation
        + OBV 7-candle trend
        + ATR relative to 7-candle mean
    Position size scales linearly with |pred-price - current_price|.
    """
    # --- Core settings --------------------------------------------------------
    timeframe               = '1h'
    can_short               = True          # Allow shorting
    startup_candle_count    = 200           # Indicator warmup period

    # --- Futures trading settings --------------------------------------------
    trading_mode            = 'futures'     # Set to futures mode
    margin_mode             = 'isolated'     # Set to isolated margin mode
    default_leverage        = 1.0            # Set default leverage to 1, equivalent to no leverage

    # --- Risk / money management ---------------------------------------------
    minimal_roi             = {"0": 0.0}
    stoploss                = -0.99
    process_only_new_candles = True

    # --- Money management settings --------------------------------------------
    stake_amount            = 100            # Fixed stake amount per trade
    position_staking        = False          # Disable position staking
    max_open_trades         = 3              # Maximum number of open trades

    # --- Trading settings -----------------------------------------------------
    use_exit_signal         = True           # Use exit signal
    exit_profit_only        = False          # Exit only when profitable
    ignore_roi_if_entry_signal = True        # Ignore ROI if entry signal
    position_adjustment_enable = False       # Disable position adjustment

    long_threshold          = 0.002         # +0.2 %
    short_threshold         = -0.002        # −0.2 %
    max_prediction_deviation = 0.04        # Maximum prediction deviation 4%
    scale_return            = 0.01           # 1 % move ⇒ full stake
    max_position_pct        = 1.0            # 100 % wallet stake cap

    # --- Model + scaler paths -------------------------------------------------
    model_path              = 'user_data/models/lstm/price_lstm.pt'
    feat_scaler_path        = 'user_data/models/lstm/feature_scaler.joblib'
    targ_scaler_path        = 'user_data/models/lstm/target_scaler.joblib'

    # --- Feature list (must match trainer) ------------------------------------
    feature_cols: List[str] = [
        'close', 'open', 'high', 'low', 'volume',
        'return_1', 'return_3', 'return_6',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volatility_24', 'obv', 'mom_10'
    ]
    WINDOW_SIZE = 24                      # must match trainer

    # -------------------------------------------------------------------------
    # Init – load model / scalers once
    # -------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.x_scaler = joblib.load(self.feat_scaler_path)
        self.y_scaler = joblib.load(self.targ_scaler_path)

        self.model = PriceLSTM(input_size=len(self.feature_cols)).to(self.device)
        
        # Load model
        print("\n=== Model loading check ===")
        print(f"Model file path: {self.model_path}")
        chkpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        print(f"Checkpoint content: {chkpt.keys()}")
        
        # Check model state
        print("\n=== Model state check ===")
        print(f"Model device: {self.device}")
        print(f"Model training mode: {self.model.training}")
        
        # Load model weights
        self.model.load_state_dict(chkpt['model_state_dict'])
        self.model.eval()
        
        # Check model weights
        print("\n=== Model weights check ===")
        for name, param in self.model.named_parameters():
            print(f"\nParameter name: {name}")
            print(f"Shape: {param.shape}")
            print(f"Mean: {param.data.mean().item():.6f}")
            print(f"Standard deviation: {param.data.std().item():.6f}")
            print(f"Minimum value: {param.data.min().item():.6f}")
            print(f"Maximum value: {param.data.max().item():.6f}")
        
        # Test model output
        test_input = torch.randn(1, self.WINDOW_SIZE, len(self.feature_cols)).to(self.device)
        with torch.no_grad():
            test_output = self.model(test_input)
            print(f"\nTest input shape: {test_input.shape}")
            print(f"Test output: {test_output.item():.6f}")
        print("===================\n")

    # -------------------------------------------------------------------------
    # Indicator calculation (executed once per candle)
    # -------------------------------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['return_1'] = df['close'].pct_change()
        df['return_3'] = df['close'].pct_change(3)
        df['return_6'] = df['close'].pct_change(6)

        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'], df['bb_width'] = calculate_bollinger(
            df['close'])
        df['atr'] = calculate_atr(df)
        df['volatility_24'] = calculate_volatility(df['close'])
        df['obv'] = calculate_obv(df)
        df['mom_10'] = df['close'] - df['close'].shift(10)

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Add debug information
        print("\n=== Data check ===")
        print(f"Data length: {len(df)}")
        print(f"Data columns: {df.columns.tolist()}")
        print(f"Last 5 rows:")
        print(df.tail())
        print("================\n")

        return df

    # -------------------------------------------------------------------------
    # Internal helper – make one prediction
    # -------------------------------------------------------------------------
    def _predict_price(self, df: DataFrame) -> float:
        """Return absolute price prediction for next hour."""
        if len(df) < self.WINDOW_SIZE:
            print(f"Warning: Not enough data points. Need {self.WINDOW_SIZE}, got {len(df)}")
            return np.nan
            
        # Get the index of the current time point
        current_idx = df.index.get_loc(df.index[-1])
        if current_idx < self.WINDOW_SIZE:
            return np.nan
            
        # Use data window before current time point
        window_start = current_idx - self.WINDOW_SIZE
        window_end = current_idx
        
        # Get data window
        window = (df[self.feature_cols]
                  .iloc[window_start:window_end]  # Use data before current time point
                  .values.astype(np.float32))
        
        # Add debug information
        print(f"\n=== LSTM prediction check ===")
        print(f"Current time point: {df.index[current_idx]}")
        print(f"Data window start time: {df.index[window_start]}")
        print(f"Data window end time: {df.index[window_end-1]}")
        print(f"Window shape: {window.shape}")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Window data (last 5 rows):")
        print(window[-5:])
        
        # (24, feat) → scale → (1, 24, feat)
        scaled = self.x_scaler.transform(window).reshape(1,
                                                         self.WINDOW_SIZE,
                                                         -1)
        print(f"Scaled shape: {scaled.shape}")
        print(f"Scaled data (The last 5 lines):")
        print(scaled[0, -5:])
        
        # Check the model status
        print(f"Model device: {self.device}")
        print(f"Model training mode: {self.model.training}")
        
        with torch.no_grad():
            input_tensor = torch.tensor(scaled, device=self.device)
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Input tensor device: {input_tensor.device}")
            pred_norm = self.model(input_tensor).item()
            print(f"Raw prediction (normalized): {pred_norm}")
            
        # inverse_transform expects 2-D
        pred_price = self.y_scaler.inverse_transform([[pred_norm]])[0][0]
        print(f"Predicted price: {pred_price:.2f}")
        print("===================\n")
        
        return float(pred_price)

    # -------------------------------------------------------------------------
    # ENTRY signals
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, df: DataFrame,
                             metadata: dict) -> DataFrame:
        """
        Entry signal logic
        """
        # Get the current time point
        current_time = df.index[-1]
        current_idx = df.index.get_loc(current_time)
        
        # Ensure there is enough historical data
        if current_idx < self.WINDOW_SIZE:
            return df
            
        # Only use data before the current time point
        historical_data = df.iloc[:current_idx+1].copy()
        
        # Use historical data for prediction
        pred_price = self._predict_price(historical_data)
        if np.isnan(pred_price):
            return df
            
        df.loc[current_time, 'pred_price'] = pred_price

        cur_price = df['close'].iloc[-1]
        delta_pct = (pred_price - cur_price) / cur_price

        # Add more detailed debug information
        print(f"\n=== Trading signal check ===")
        print(f"Current time: {current_time}")
        print(f"Current price: {cur_price:.2f}")
        print(f"Predicted price: {pred_price:.2f}")
        print(f"Price deviation: {delta_pct:.4%}")
        print(f"Long threshold: {self.long_threshold:.4%}")
        print(f"Short threshold: {self.short_threshold:.4%}")
        print(f"Long condition met: {delta_pct > self.long_threshold}")
        print(f"Short condition met: {delta_pct < self.short_threshold}")
        print(f"MACD: {df['macd'].iloc[-1]:.2f}")
        print(f"MACD signal: {df['macd_signal'].iloc[-1]:.2f}")
        print(f"OBV: {df['obv'].iloc[-1]:.2f}")
        print(f"OBV(7 periods ago): {df['obv'].shift(7).iloc[-1]:.2f}")
        print("===================\n")

        # Long conditions
        long_conditions = (
            (delta_pct > self.long_threshold) &  # Predicted price higher than current price 0.2%
            (delta_pct < self.max_prediction_deviation) &  # Prediction deviation not more than 5%
            (df['macd'] > df['macd_signal']) &  # MACD golden cross
            (df['macd'] > 0) &  # MACD above zero axis
            (df['obv'] > df['obv'].shift(7)) &  # OBV upward trend
            (df['obv'] > df['obv'].shift(14)) &  # OBV medium-term upward trend
            (df['close'] > df['close'].shift(7))  # Price 7-period upward trend
        )

        # Short conditions
        short_conditions = (
            (delta_pct < self.short_threshold) &  # Predicted price lower than current price 0.2%
            (delta_pct > -self.max_prediction_deviation) &  # Prediction deviation not more than 5%
            (df['macd'] < df['macd_signal']) &  # MACD death cross
            (df['macd'] < 0) &  # MACD below zero axis
            (df['obv'] < df['obv'].shift(7)) &  # OBV downward trend
            (df['obv'] < df['obv'].shift(14)) &  # OBV medium-term downward trend
            (df['close'] < df['close'].shift(7))  # Price 7-period downward trend
        )

        # Set entry signals
        df.loc[long_conditions, 'enter_long'] = 1
        df.loc[short_conditions, 'enter_short'] = 1

        # Print signal status
        print(f"Long signal: {df.loc[current_time, 'enter_long'] if 'enter_long' in df.columns else 0}")
        print(f"Short signal: {df.loc[current_time, 'enter_short'] if 'enter_short' in df.columns else 0}")
        return df

    # -------------------------------------------------------------------------
    # EXIT signals  —— Direction reversal is the same as closing
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, df: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Exit signal logic
        """
        if 'pred_price' not in df.columns:
            return df
        last = df.index[-1]
        pred_price = df['pred_price'].iloc[-1]
        if np.isnan(pred_price):
            return df

        cur_price = df['close'].iloc[-1]
        delta_pct = (pred_price - cur_price) / cur_price

        # Long exit conditions
        exit_long_conditions = (
            (delta_pct < 0) |  # Predicted price lower than current price
            (df['macd'] < df['macd_signal']) |  # MACD death cross
            (df['obv'] < df['obv'].shift(7))  # OBV downward trend
        )

        # Short exit conditions
        exit_short_conditions = (
            (delta_pct > 0) |  # Predicted price higher than current price
            (df['macd'] > df['macd_signal']) |  # MACD golden cross
            (df['obv'] > df['obv'].shift(7))  # OBV upward trend
        )

        # Set exit signals
        df.loc[exit_long_conditions, 'exit_long'] = 1
        df.loc[exit_short_conditions, 'exit_short'] = 1

        return df
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Set leverage
        """
        return self.default_leverage
