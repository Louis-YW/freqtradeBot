"""LSTM regression model to predict next‑day BTC/USDT close price with sentiment indicators.

Run
---
python lstm_price_regression.py --data_path path/to/BTC_1d.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Technical‑indicator helpers
# -----------------------------------------------------------------------------

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger(prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma
    return upper, lower, width


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def calculate_volatility(prices: pd.Series, window: int = 24) -> pd.Series:
    return prices.pct_change().rolling(window).std()

# -----------------------------------------------------------------------------
# Dataset & Model
# -----------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 添加批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(32)
        
        # 修改全连接层
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
        
        # 初始化权重
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
        # LSTM层
        out, _ = self.lstm(x)
        
        # 获取最后一个时间步的输出
        last_hidden = out[:, -1]
        
        # 应用批归一化
        last_hidden = self.bn1(last_hidden)
        
        # 全连接层
        return self.fc(last_hidden)

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class LSTMTrainer:
    def __init__(self, data_path: str | Path, sentiment_path: str | Path, model_dir: str | Path = 'ft_userdata/user_data/models/lstm_with_sentiment', window_size: int = 10, horizon: int = 1):
        self.data_path = Path(data_path)
        self.sentiment_path = Path(sentiment_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols: List[str] = []
        self.sentiment_cols = [
            'emo_vol_resonance', 'signal_strength_ratio', 'pca1', 'rhythm_sync',
            'dual_momentum_sq', 'dual_momentum', 'momentum_diff', 'sentiment_shock',
            'pca2', 'sentiment_snr', 'weighted_sentiment', 'extreme_resonance_strength'
        ]

    # ---------------- Data ----------------
    def load_raw(self) -> pd.DataFrame:
        # 加载技术指标数据
        with open(self.data_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')
        
        # 加载情绪指标数据
        sent_df = pd.read_csv(self.sentiment_path)
        sent_df['date'] = pd.to_datetime(sent_df['created_time'])
        sent_df = sent_df.set_index('date')
        
        # 设置时间范围过滤
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2024-06-30')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        sent_df = sent_df[(sent_df.index >= start_date) & (sent_df.index <= end_date)]
        
        # 计算情绪指标
        N = 20  # 用于计算移动平均的窗口大小
        k = 5   # 用于计算动量的窗口大小
        
        # 情绪冲击度
        sent_df['sentiment_mean'] = sent_df['sentiment_score'].rolling(N).mean()
        sent_df['sentiment_std'] = sent_df['sentiment_score'].rolling(N).std()
        sent_df['sentiment_shock'] = (sent_df['sentiment_score'] - sent_df['sentiment_mean']) / sent_df['sentiment_std']
        
        # 情绪与成交量共振
        df['volume_ma'] = df['volume'].rolling(N).mean()
        sent_df['emo_vol_resonance'] = sent_df['sentiment_score'] * (df['volume_ma'] / df['volume'])
        
        # Awareness 因子
        sent_df['sentiment_snr'] = sent_df['sentiment_score'].abs() / sent_df['sentiment_std']
        sent_df['weighted_sentiment'] = sent_df['sentiment_score'] * (df['volume'] / df['volume_ma'])
        
        # Dual momentum
        df['price_mom'] = df['close'].pct_change(k)
        sent_df['sent_mom'] = sent_df['sentiment_score'].diff(k)
        sent_df['dual_momentum'] = df['price_mom'] * sent_df['sent_mom']
        sent_df['dual_momentum_sq'] = sent_df['dual_momentum'] ** 2
        sent_df['momentum_diff'] = sent_df['sent_mom'] - df['price_mom']
        
        # Extreme resonance
        delta = df['close'].diff()
        gain = delta.where(delta>0, 0).rolling(14).mean()
        loss = -delta.where(delta<0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - 100 / (1 + rs)
        low_q = sent_df['sentiment_score'].quantile(0.1)
        high_q = sent_df['sentiment_score'].quantile(0.9)
        sent_df['extreme_resonance'] = (
            ((sent_df['sentiment_score'] <= low_q) & (df['rsi'] < 30)) |
            ((sent_df['sentiment_score'] >= high_q) & (df['rsi'] > 70))
        )
        
        # 修复extreme_resonance_strength的计算
        sent_df['extreme_resonance_strength'] = 0.0
        mask = sent_df['extreme_resonance']
        if mask.any():
            # 计算情绪强度部分
            sentiment_strength = ((sent_df.loc[mask, 'sentiment_score'] - high_q) / 
                                (sent_df['sentiment_score'].quantile(0.99) - high_q)).fillna(0)
            # 计算RSI强度部分
            rsi_strength = ((df.loc[mask.index, 'rsi'] - 70) / 30).fillna(0)
            # 合并两部分
            sent_df.loc[mask, 'extreme_resonance_strength'] = sentiment_strength + rsi_strength
        
        # Rhythm sync
        df['volume_change'] = df['volume'].pct_change()
        sent_df['sentiment_change'] = sent_df['sentiment_score'].diff()
        sent_df['rhythm_sync'] = df['volume_change'].rolling(N).corr(sent_df['sentiment_change'])
        
        # PCA 因子
        features = [
            'price_mom', 'sent_mom', 'dual_momentum', 'dual_momentum_sq', 'momentum_diff',
            'sentiment_shock', 'emo_vol_resonance', 'sentiment_snr', 'weighted_sentiment',
            'extreme_resonance_strength', 'rhythm_sync'
        ]
        X = pd.concat([df[['price_mom']], sent_df[features[1:]]], axis=1).dropna()
        pca = PCA(n_components=2)
        comp = pca.fit_transform(X)
        sent_df.loc[X.index, 'pca1'] = comp[:,0]
        sent_df.loc[X.index, 'pca2'] = comp[:,1]
        
        # Signal strength ratio
        sent_df['signal_strength_ratio'] = df['price_mom'].diff() / sent_df['sentiment_score'].diff()
        
        # 合并数据
        # 确保两个数据集的时间范围一致
        common_dates = df.index.intersection(sent_df.index)
        df = df.loc[common_dates]
        sent_df = sent_df.loc[common_dates]
        
        df = pd.merge_asof(
            df.sort_index(),
            sent_df[self.sentiment_cols].sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        
        return df

    def prepare_data(self) -> pd.DataFrame:
        df = self.load_raw()
        
        # 技术指标
        df['return_1'] = df['close'].pct_change()
        df['return_3'] = df['close'].pct_change(3)
        df['return_6'] = df['close'].pct_change(6)
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'], df['bb_width'] = calculate_bollinger(df['close'])
        df['atr'] = calculate_atr(df)
        df['volatility_24'] = calculate_volatility(df['close'])
        df['obv'] = calculate_obv(df)
        df['mom_10'] = df['close'] - df['close'].shift(10)
        
        # 合并所有特征列
        self.feature_cols = [
            'close', 'open', 'high', 'low', 'volume',
            'return_1', 'return_3', 'return_6', 'rsi', 'macd',
            'macd_signal', 'bb_upper', 'bb_lower', 'bb_width',
            'atr', 'volatility_24', 'obv', 'mom_10'
        ] + self.sentiment_cols
        
        return df.ffill().bfill()

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.dropna(subset=self.feature_cols)
        X, y = [], []
        for i in range(self.window_size, len(df) - self.horizon):
            X.append(df[self.feature_cols].iloc[i-self.window_size:i].values)
            y.append(df['close'].iloc[i + self.horizon])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    # ---------------- Train ----------------
    def train(self, epochs: int = 100, batch_size: int = 64, validation_split: float = 0.2, patience: int = 15, lr: float = 1e-3):
        df = self.prepare_data()
        X, y = self.create_sequences(df)
        n_train = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:n_train], X[n_train:]; y_train, y_val = y[:n_train], y[n_train:]

        feat_scaler, targ_scaler = StandardScaler(), StandardScaler()
        X_train_resh, X_val_resh = X_train.reshape(-1, X.shape[-1]), X_val.reshape(-1, X.shape[-1])
        feat_scaler.fit(X_train_resh)
        X_train = feat_scaler.transform(X_train_resh).reshape(X_train.shape)
        X_val = feat_scaler.transform(X_val_resh).reshape(X_val.shape)
        targ_scaler.fit(y_train)
        y_train, y_val = targ_scaler.transform(y_train), targ_scaler.transform(y_val)

        joblib.dump(feat_scaler, self.model_dir / 'feature_scaler.joblib')
        joblib.dump(targ_scaler, self.model_dir / 'target_scaler.joblib')

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, criterion = PriceLSTM(len(self.feature_cols)).to(device), nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )

        # 添加训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_pred_mean': [],
            'train_pred_std': [],
            'val_pred_mean': [],
            'val_pred_std': []
        }

        best_val, no_improv = float('inf'), 0
        epoch_iter = tqdm(range(1, epochs+1), desc='Epoch', unit='epoch')
        
        for epoch in epoch_iter:
            # ---- Train ----
            model.train()
            train_loss = 0.0
            train_preds = []
            
            for xb, yb in tqdm(train_loader, desc='Train batches', leave=False):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                train_preds.extend(pred.detach().cpu().numpy())
            
            train_loss /= len(train_loader)
            train_preds = np.array(train_preds)
            
            # ---- Validate ----
            model.eval()
            val_loss = 0.0
            val_preds = []
            
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc='Val batches', leave=False):
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item()
                    val_preds.extend(pred.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_preds = np.array(val_preds)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录预测统计信息
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_pred_mean'].append(train_preds.mean())
            history['train_pred_std'].append(train_preds.std())
            history['val_pred_mean'].append(val_preds.mean())
            history['val_pred_std'].append(val_preds.std())
            
            # 打印详细信息
            epoch_iter.set_postfix({
                "train_mse": f"{train_loss:.5f}",
                "val_mse": f"{val_loss:.5f}",
                "train_pred_mean": f"{train_preds.mean():.3f}",
                "train_pred_std": f"{train_preds.std():.3f}",
                "val_pred_mean": f"{val_preds.mean():.3f}",
                "val_pred_std": f"{val_preds.std():.3f}"
            })

            # ---- Early stopping ----
            if val_loss < best_val - 1e-4:
                best_val, no_improv = val_loss, 0
                # 保存完整的模型状态
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mse': val_loss,
                    'history': history,
                    'feature_cols': self.feature_cols,
                    'window_size': self.window_size,
                    'horizon': self.horizon
                }, self.model_dir / 'price_lstm_with_sentiment.pt')
                
                # 打印模型参数统计信息
                print("\n模型参数统计:")
                for name, param in model.named_parameters():
                    print(f"{name}:")
                    print(f"  均值: {param.data.mean().item():.6f}")
                    print(f"  标准差: {param.data.std().item():.6f}")
                    print(f"  最小值: {param.data.min().item():.6f}")
                    print(f"  最大值: {param.data.max().item():.6f}")
            else:
                no_improv += 1
                if no_improv >= patience:
                    print(f"Early stopping: no improvement {patience} epochs.")
                    break

        print("Training finished. Best val MSE:", best_val)
        return model

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/Users/yichenwu/Documents/DIAcoursework/ft_userdata/user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.json')
    parser.add_argument('--sentiment_path', default='/Users/yichenwu/Documents/DIAcoursework/ft_userdata/user_data/data/processed/sentiment_features_1D.csv')
    parser.add_argument('--model_dir', default='ft_userdata/user_data/models/lstm_with_sentiment')
    args = parser.parse_args()
    trainer = LSTMTrainer(args.data_path, args.sentiment_path, args.model_dir)
    trainer.train()

if __name__ == '__main__':
    main()
