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
class TraditionalAggressiveStrategy(IStrategy):
    """
    This is an aggressive cryptocurrency trading strategy based on the following principles:
    1. Quickly capture trend changes
    2. Use multiple technical indicators to confirm trends
    3. Enter positions early in the trend
    4. Adjust positions promptly based on technical indicator changes
    
    Strategy features:
    - Use EMA system for quick response to price changes
    - Use RSI and MACD to capture overbought/oversold conditions
    - Use Bollinger Bands to judge price volatility
    - Use volume to confirm trends
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Allow shorting
    can_short: bool = True

    # Minimum ROI settings - lower profit targets for faster exits
    minimal_roi = {
        "60": 0.02,   # Exit with 2% profit after 1 hour
        "30": 0.03,   # Exit with 3% profit after 30 minutes
        "15": 0.04,   # Exit with 4% profit after 15 minutes
        "0": 0.05     # Exit immediately with 5% profit
    }

    # Stop loss settings - tighter stops to control risk
    stoploss = -0.03  # 3% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1% trailing stop
    trailing_stop_positive_offset = 0.02  # 2% offset
    trailing_only_offset_is_reached = True

    # Use 15-minute timeframe
    timeframe = "15m"

    # Only run on new candles
    process_only_new_candles = True

    # These values can be overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles needed for strategy
    startup_candle_count: int = 100

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
        # Calculate EMA system
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Calculate MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Calculate Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        # Calculate volume changes
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Calculate price momentum
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=10)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal logic - quickly capture trend changes
        """
        dataframe.loc[
            (
                # Long conditions - any condition can trigger
                (
                    # Condition 1: EMA system bullish alignment
                    (dataframe['close'] > dataframe['ema20']) &
                    (dataframe['ema20'] > dataframe['ema50']) &
                    (dataframe['ema50'] > dataframe['ema100'])
                ) |
                (
                    # Condition 2: RSI oversold bounce
                    (dataframe['rsi'] < 30) &
                    (dataframe['rsi'].shift(1) < dataframe['rsi'])
                ) |
                (
                    # Condition 3: MACD golden cross
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
                ) |
                (
                    # Condition 4: Bollinger Band lower band bounce
                    (dataframe['close'] < dataframe['bb_lowerband']) &
                    (dataframe['close'].shift(1) < dataframe['close'])
                ) |
                (
                    # Condition 5: Volume breakout
                    (dataframe['volume_ratio'] > 2) &
                    (dataframe['close'] > dataframe['ema20']) &
                    (dataframe['momentum'] > 0)
                )
            ),
            'enter_long'
        ] = 1

        dataframe.loc[
            (
                # Short conditions - any condition can trigger
                (
                    # Condition 1: EMA system bearish alignment
                    (dataframe['close'] < dataframe['ema20']) &
                    (dataframe['ema20'] < dataframe['ema50']) &
                    (dataframe['ema50'] < dataframe['ema100'])
                ) |
                (
                    # Condition 2: RSI overbought pullback
                    (dataframe['rsi'] > 70) &
                    (dataframe['rsi'].shift(1) > dataframe['rsi'])
                ) |
                (
                    # Condition 3: MACD death cross
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
                ) |
                (
                    # Condition 4: Bollinger Band upper band pullback
                    (dataframe['close'] > dataframe['bb_upperband']) &
                    (dataframe['close'].shift(1) > dataframe['close'])
                ) |
                (
                    # Condition 5: Volume breakdown
                    (dataframe['volume_ratio'] > 2) &
                    (dataframe['close'] < dataframe['ema20']) &
                    (dataframe['momentum'] < 0)
                )
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal logic - quick response to trend changes
        """
        dataframe.loc[
            (
                # Close long conditions - any condition can trigger
                (dataframe['close'] < dataframe['ema20']) |  # Price breaks below 20 EMA
                (dataframe['rsi'] > 80) |                    # RSI overbought
                (dataframe['macd'] < dataframe['macdsignal']) |  # MACD death cross
                (dataframe['close'] > dataframe['bb_upperband']) |  # Price breaks above upper Bollinger Band
                (dataframe['momentum'] < 0)                 # Momentum turns negative
            ),
            'exit_long'
        ] = 1

        dataframe.loc[
            (
                # Close short conditions - any condition can trigger
                (dataframe['close'] > dataframe['ema20']) |  # Price breaks above 20 EMA
                (dataframe['rsi'] < 20) |                    # RSI oversold
                (dataframe['macd'] > dataframe['macdsignal']) |  # MACD golden cross
                (dataframe['close'] < dataframe['bb_lowerband']) |  # Price breaks below lower Bollinger Band
                (dataframe['momentum'] > 0)                 # Momentum turns positive
            ),
            'exit_short'
        ] = 1

        return dataframe
