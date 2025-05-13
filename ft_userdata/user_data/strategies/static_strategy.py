# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# This class is a sample. Feel free to customize it.
class StaticStrategy(IStrategy):
    """
    A conservative trading strategy based on multiple technical indicator confirmations:
    1. Long-term trading (1-12 weeks): Requires continuous technical indicator confirmation for a week
    2. Medium-term trading (4-7 days): Based on daily average technical indicators
    
    Strategy features:
    - Uses multiple technical indicators for cross-validation
    - Requires continuous indicator confirmation for trading
    - Implements conservative risk management
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Allow shorting
    can_short: bool = True

    # More conservative ROI settings
    minimal_roi = {
        "720": 0.05,   # Exit with 5% profit after 30 days
        "480": 0.10,   # Exit with 10% profit after 20 days
        "240": 0.15,   # Exit with 15% profit after 10 days
        "120": 0.20,   # Exit with 20% profit after 5 days
        "0": 0.25      # Exit immediately with 25% profit
    }

    # More conservative stop loss settings
    stoploss = -0.10  # 10% stop loss

    # Trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.05  # 5% trailing stop
    trailing_stop_positive_offset = 0.07  # 7% offset
    trailing_only_offset_is_reached = True

    # Use daily timeframe
    timeframe = "1d"

    # Only run on new candles
    process_only_new_candles = True

    # These values can be overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Need more historical data for indicator calculation
    startup_candle_count: int = 200

    # Order type settings
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Order time in force
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all required technical indicators
        """
        # Basic price changes
        dataframe['price_change'] = dataframe['close'].pct_change().fillna(0)
        dataframe['close_open_ratio'] = (dataframe['close'] / dataframe['open']).fillna(1)
        
        # Moving averages
        for n in [5, 10, 20, 50, 100]:
            dataframe[f'ma{n}'] = ta.SMA(dataframe, timeperiod=n)
        
        # Volatility
        dataframe['volatility20'] = dataframe['close'].rolling(20, min_periods=1).std().fillna(0)
        
        # ATR
        high_low = dataframe['high'] - dataframe['low']
        high_cp = (dataframe['high'] - dataframe['close'].shift()).abs()
        low_cp = (dataframe['low'] - dataframe['close'].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        dataframe['atr14'] = tr.rolling(14, min_periods=1).mean().fillna(0)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd_line'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        
        # Bollinger bandwidth
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bollinger_bw'] = ((bollinger['upper'] - bollinger['lower']) / bollinger['mid']).fillna(0)
        
        # OBV
        dataframe['obv'] = (np.sign(dataframe['close'].diff().fillna(0)) * dataframe['volume']).cumsum().fillna(0)
        
        # Calculate 7-day moving averages
        for col in ['ma5', 'ma10', 'ma20', 'ma50', 'ma100', 'obv', 'atr14']:
            dataframe[f'{col}_7d_avg'] = dataframe[col].rolling(7, min_periods=1).mean()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal logic
        """
        # Long-term trading conditions (1-12 weeks)
        long_term_long_conditions = (
            (dataframe['close_open_ratio'] > 1.02) &  # Close price significantly higher than open
            (dataframe['obv'] > dataframe['obv'].shift(7)) &  # OBV uptrend
            (dataframe['volatility20'] < dataframe['volatility20'].rolling(7).mean()) &  # Decreasing volatility
            (dataframe['macd_line'] > dataframe['macd_signal']) &  # MACD golden cross
            (dataframe['ma5'] > dataframe['ma10']) &  # Short-term MA above long-term MA
            (dataframe['price_change'] > 0) &  # Price increase
            (dataframe['bollinger_bw'] < dataframe['bollinger_bw'].rolling(7).mean()) &  # Bollinger bands narrowing
            (dataframe['atr14'] < dataframe['atr14'].rolling(7).mean())  # Decreasing ATR
        )

        long_term_short_conditions = (
            (dataframe['close_open_ratio'] < 0.98) &  # Close price significantly lower than open
            (dataframe['obv'] < dataframe['obv'].shift(7)) &  # OBV downtrend
            (dataframe['volatility20'] > dataframe['volatility20'].rolling(7).mean()) &  # Increasing volatility
            (dataframe['macd_line'] < dataframe['macd_signal']) &  # MACD death cross
            (dataframe['ma5'] < dataframe['ma10']) &  # Short-term MA below long-term MA
            (dataframe['price_change'] < 0) &  # Price decrease
            (dataframe['bollinger_bw'] > dataframe['bollinger_bw'].rolling(7).mean()) &  # Bollinger bands expanding
            (dataframe['atr14'] > dataframe['atr14'].rolling(7).mean())  # Increasing ATR
        )

        # Medium-term trading conditions (4-7 days)
        medium_term_long_conditions = (
            (dataframe['ma5_7d_avg'] > dataframe['ma10_7d_avg']) &
            (dataframe['ma10_7d_avg'] > dataframe['ma20_7d_avg']) &
            (dataframe['ma20_7d_avg'] > dataframe['ma50_7d_avg']) &
            (dataframe['ma50_7d_avg'] > dataframe['ma100_7d_avg']) &
            (dataframe['obv_7d_avg'] > dataframe['obv_7d_avg'].shift(1)) &
            (dataframe['atr14_7d_avg'] < dataframe['atr14_7d_avg'].rolling(7).mean())
        )

        medium_term_short_conditions = (
            (dataframe['ma5_7d_avg'] < dataframe['ma10_7d_avg']) &
            (dataframe['ma10_7d_avg'] < dataframe['ma20_7d_avg']) &
            (dataframe['ma20_7d_avg'] < dataframe['ma50_7d_avg']) &
            (dataframe['ma50_7d_avg'] < dataframe['ma100_7d_avg']) &
            (dataframe['obv_7d_avg'] < dataframe['obv_7d_avg'].shift(1)) &
            (dataframe['atr14_7d_avg'] > dataframe['atr14_7d_avg'].rolling(7).mean())
        )

        # Set entry signals
        dataframe.loc[long_term_long_conditions, 'enter_long'] = 1
        dataframe.loc[long_term_short_conditions, 'enter_short'] = 1
        dataframe.loc[medium_term_long_conditions, 'enter_long'] = 1
        dataframe.loc[medium_term_short_conditions, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal logic
        """
        # Long-term trading exit conditions
        long_term_exit_long = (
            (dataframe['close_open_ratio'] < 0.99) |
            (dataframe['obv'] < dataframe['obv'].shift(7)) |
            (dataframe['macd_line'] < dataframe['macd_signal']) |
            (dataframe['ma5'] < dataframe['ma10'])
        )

        long_term_exit_short = (
            (dataframe['close_open_ratio'] > 1.01) |
            (dataframe['obv'] > dataframe['obv'].shift(7)) |
            (dataframe['macd_line'] > dataframe['macd_signal']) |
            (dataframe['ma5'] > dataframe['ma10'])
        )

        # Medium-term trading exit conditions
        medium_term_exit_long = (
            (dataframe['ma5_7d_avg'] < dataframe['ma10_7d_avg']) |
            (dataframe['obv_7d_avg'] < dataframe['obv_7d_avg'].shift(1))
        )

        medium_term_exit_short = (
            (dataframe['ma5_7d_avg'] > dataframe['ma10_7d_avg']) |
            (dataframe['obv_7d_avg'] > dataframe['obv_7d_avg'].shift(1))
        )

        # Set exit signals
        dataframe.loc[long_term_exit_long | medium_term_exit_long, 'exit_long'] = 1
        dataframe.loc[long_term_exit_short | medium_term_exit_short, 'exit_short'] = 1

        return dataframe
