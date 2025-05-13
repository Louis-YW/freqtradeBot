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
class TraditionalConservativeStrategy(IStrategy):
    """
    This is an extremely conservative cryptocurrency trading strategy based on the following principles:
    1. Only trade when the trend is extremely clear
    2. Use multiple moving averages to confirm the trend
    3. Require trend continuity and strength
    4. Rather miss opportunities than make uncertain trades
    
    Strategy features:
    - Use 200-day moving average as the main trend indicator
    - Use 50-day and 20-day moving averages as auxiliary confirmation
    - Require perfect bullish/bearish alignment of moving averages
    - Require sufficient trend continuity and strength
    - Significantly reduce trading frequency, only trade at the most certain times
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Allow shorting
    can_short: bool = True

    # Minimum ROI settings - increase profit targets
    minimal_roi = {
        "480": 0.10,  # Exit with 10% profit after 20 hours
        "240": 0.15,  # Exit with 15% profit after 10 hours
        "120": 0.20,  # Exit with 20% profit after 5 hours
        "0": 0.25     # Exit immediately with 25% profit
    }

    # Stop loss settings
    stoploss = -0.15  # 15% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.08  # 8% trailing stop
    trailing_stop_positive_offset = 0.10  # 10% offset
    trailing_only_offset_is_reached = True

    # Use daily timeframe
    timeframe = "1d"

    # Only run on new candles
    process_only_new_candles = True

    # These values can be overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles needed for strategy
    startup_candle_count: int = 200

    # Order types
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
        Calculate technical indicators
        """
        # Calculate long-term moving averages
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        
        # Calculate moving average slopes (for trend strength)
        dataframe['sma200_slope'] = (dataframe['sma200'] - dataframe['sma200'].shift(20)) / dataframe['sma200'].shift(20) * 100
        dataframe['sma50_slope'] = (dataframe['sma50'] - dataframe['sma50'].shift(10)) / dataframe['sma50'].shift(10) * 100
        dataframe['sma20_slope'] = (dataframe['sma20'] - dataframe['sma20'].shift(5)) / dataframe['sma20'].shift(5) * 100

        # Calculate trend continuity indicator
        dataframe['trend_strength'] = (
            (dataframe['sma20'] > dataframe['sma50']).astype(int) +
            (dataframe['sma50'] > dataframe['sma200']).astype(int) +
            (dataframe['sma20_slope'] > 0).astype(int) +
            (dataframe['sma50_slope'] > 0).astype(int) +
            (dataframe['sma200_slope'] > 0).astype(int)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal logic - only trade when trend is extremely clear
        """
        dataframe.loc[
            (
                # Long conditions - all must be met
                (dataframe['close'] > dataframe['sma200']) &  # Price above 200-day MA
                (dataframe['close'] > dataframe['sma50']) &   # Price above 50-day MA
                (dataframe['close'] > dataframe['sma20']) &   # Price above 20-day MA
                (dataframe['sma20'] > dataframe['sma50']) &   # 20-day MA above 50-day MA
                (dataframe['sma50'] > dataframe['sma200']) &  # 50-day MA above 200-day MA
                (dataframe['sma200_slope'] > 1.0) &           # 200-day MA clearly upward sloping
                (dataframe['sma50_slope'] > 2.0) &            # 50-day MA clearly upward sloping
                (dataframe['sma20_slope'] > 3.0) &            # 20-day MA clearly upward sloping
                (dataframe['trend_strength'] >= 4) &          # Trend strength indicator at least 4 points
                (dataframe['volume'] > 0)                     # Ensure there is volume
            ),
            'enter_long'
        ] = 1

        dataframe.loc[
            (
                # Short conditions - all must be met
                (dataframe['close'] < dataframe['sma200']) &  # Price below 200-day MA
                (dataframe['close'] < dataframe['sma50']) &   # Price below 50-day MA
                (dataframe['close'] < dataframe['sma20']) &   # Price below 20-day MA
                (dataframe['sma20'] < dataframe['sma50']) &   # 20-day MA below 50-day MA
                (dataframe['sma50'] < dataframe['sma200']) &  # 50-day MA below 200-day MA
                (dataframe['sma200_slope'] < -1.0) &          # 200-day MA clearly downward sloping
                (dataframe['sma50_slope'] < -2.0) &           # 50-day MA clearly downward sloping
                (dataframe['sma20_slope'] < -3.0) &           # 20-day MA clearly downward sloping
                (dataframe['trend_strength'] <= 1) &          # Trend strength indicator at most 1 point
                (dataframe['volume'] > 0)                     # Ensure there is volume
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal logic - exit when trend starts to weaken
        """
        dataframe.loc[
            (
                # Close long conditions
                (dataframe['close'] < dataframe['sma20']) |  # Price breaks below 20-day MA
                (dataframe['sma20_slope'] < 0) |             # 20-day MA starts sloping downward
                (dataframe['trend_strength'] < 3)            # Trend strength weakens
            ),
            'exit_long'
        ] = 1

        dataframe.loc[
            (
                # Close short conditions
                (dataframe['close'] > dataframe['sma20']) |  # Price breaks above 20-day MA
                (dataframe['sma20_slope'] > 0) |             # 20-day MA starts sloping upward
                (dataframe['trend_strength'] > 2)            # Trend strength increases
            ),
            'exit_short'
        ] = 1

        return dataframe
