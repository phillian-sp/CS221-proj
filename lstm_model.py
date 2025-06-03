"""
lstm_rolling_predictions.py
===========================
Train an LSTM network on the aggregated 30‑minute demand series and
benchmark rolling‑origin forecasts against the same step‑sizes used in
`sarimax_rolling_predictions.py`.

Usage
-----
$ python lstm_model.py \
        --lookback 48 --horizon 6 \
        --step-sizes 1 3 6 12 24 \
        --epochs 150 --batch-size 32

The script will create a results directory named
`results-lstm-L{lookback}-H{horizon}` containing
 • the trained model (`model.pt`),
 • predictions and metrics CSV files,
 • optional plots (if matplotlib available).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM rolling‑origin forecasting")
    p.add_argument("--data-path", default="data/all_usage_halfhour.csv",
                   help="Path to aggregated network‑level CSV (default: %(default)s)")
    p.add_argument("--lookback", type=int, default=48,
                   help="Number of half‑hour slots fed into the network (default: one day = 48)")
    p.add_argument("--horizon", type=int, default=6,
                   help="Number of half‑hour slots predicted in one forward pass (default: 6 → 3 h ahead)")
    p.add_argument("--step-sizes", type=int, nargs="+", default=[1, 3, 6, 12, 24],
                   help="Rolling‑origin step sizes to evaluate (default: %(default)s)")
    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden units (default: %(default)s)")
    p.add_argument("--layers", type=int, default=2, help="Number of LSTM layers (default: %(default)s)")
    p.add_argument("--dropout", type=float, default=0.2, help="Drop‑out rate (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=10, help="Early‑stopping patience (default: %(default)s)")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Return sliding (lookback, horizon) windows as tensors."""

    def __init__(self, series: np.ndarray, lookback: int, horizon: int):
        self.lookback = lookback
        self.horizon = horizon
        X = []
        Y = []
        end = len(series) - horizon
        for i in range(lookback, end):
            X.append(series[i - lookback:i])
            Y.append(series[i:i + horizon])
        self.X = torch.from_numpy(np.stack(X)).float().unsqueeze(-1)  # (N,L,1)
        self.Y = torch.from_numpy(np.stack(Y)).float()               # (N,H)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_feats: int, hidden: int, num_layers: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, num_layers,
                             dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon)
        )

    def forward(self, x):  # x: (B,L,n_feats)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # (B,horizon)

# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, crit, optim, device):
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        optim.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def eval_epoch(model, loader, crit, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = crit(pred, y)
            total += loss.item() * len(x)
            count += len(x)
    return total / count

# -----------------------------------------------------------------------------
# Forecasting utilities
# -----------------------------------------------------------------------------

def make_rolling_predictions(model: nn.Module, series: np.ndarray, lookback: int,
                             step: int, horizon: int, scaler: Tuple[float, float], device) -> np.ndarray:
    """Generate rolling‑origin predictions for the *test* part of `series`.

    `series` must be scaled (zero mean, unit var).  We feed the most recent
    `lookback` values, predict `horizon`, then commit only the first `step`
    predictions, move the window, and repeat until the test segment is
    exhausted.
    """
    mu, std = scaler

    model.eval()
    test_len = len(series)
    preds = np.zeros(test_len)
    working_series = series.copy()  # Create a copy to work with

    # Start from the beginning to include predictions for first lookback points
    t = 0
    while t < test_len:
        # Calculate how many predictions we need to make
        remaining = min(step, test_len - t)
        current_preds = []
        
        # Make predictions in chunks of size horizon
        for i in range(0, remaining, horizon):
            chunk_size = min(horizon, remaining - i)
            
            # For the first lookback points, we need to use the training data
            if t + i < lookback:
                # Use the actual values for the first lookback points
                x = torch.from_numpy(working_series[t + i:t + i + lookback]).float().unsqueeze(0).unsqueeze(-1).to(device)
            else:
                x = torch.from_numpy(working_series[t + i - lookback:t + i]).float().unsqueeze(0).unsqueeze(-1).to(device)
                
            with torch.no_grad():
                y_hat = model(x).cpu().numpy().flatten()
            # De‑scale
            y_hat = y_hat * std + mu
            current_preds.extend(y_hat[:chunk_size])
            
            # Update working series with predictions for next iteration
            if i + horizon < remaining:
                scaled_pred = (y_hat[:chunk_size] - mu) / std
                working_series = np.concatenate([working_series, scaled_pred])
        
        # Store predictions
        preds[t:t + remaining] = current_preds
        t += step
        
    return preds

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err / true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_daily_predictions(test_data: pd.Series, predictions: pd.Series, n_steps: int, plots_dir: Path):
    """Plot predictions vs actual for each day in a 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Split into days
    days = pd.Timedelta(days=1)
    start_time = test_data.index[0]

    for day in range(6):
        ax = axes[day]
        day_start = start_time + day * days
        day_end = day_start + days

        # Plot actual and predicted
        ax.plot(test_data[day_start:day_end],
                label='Actual', linewidth=2)
        ax.plot(predictions[day_start:day_end], '--',
                label=f'{n_steps}-step Predictions', linewidth=2)

        ax.set_title(f'Day {day + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage')
        ax.legend()
        ax.grid(True)

    plt.suptitle(f'Daily LSTM Predictions with {n_steps}-step Updates', y=1.02)
    plt.tight_layout()

    # Save plot
    plt.savefig(plots_dir / f'daily_predictions_{n_steps}step.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_predictions(test_data: pd.Series, all_predictions: dict, plots_dir: Path):
    """Plot all predictions vs actual for each day in a 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Split into days
    days = pd.Timedelta(days=1)
    start_time = test_data.index[0]

    for day in range(6):
        ax = axes[day]
        day_start = start_time + day * days
        day_end = day_start + days

        # Plot actual
        ax.plot(test_data[day_start:day_end],
                label='Actual', linewidth=2)

        # Plot each prediction
        for n_steps, pred in all_predictions.items():
            ax.plot(pred[day_start:day_end], '--',
                    label=f'{n_steps}-step', linewidth=1.5, alpha=0.7)

        ax.set_title(f'Day {day + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Daily LSTM Predictions Comparison', y=1.02)
    plt.tight_layout()

    # Save plot
    plt.savefig(plots_dir / 'aggregated_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(step_sizes: List[int], all_metrics: dict, plots_dir: Path):
    """Plot and save metrics comparison."""
    plt.figure(figsize=(12, 6))
    for metric in ['MAE', 'RMSE', 'MAPE']:
        plt.plot(step_sizes, [all_metrics[n][metric] for n in step_sizes],
                 marker='o', label=metric)
    plt.xlabel('Prediction Steps')
    plt.ylabel('Error')
    plt.title('LSTM Error Metrics vs Prediction Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    raw = pd.read_csv(args.data_path, parse_dates=["time"])
    y_series = raw.drop(columns=["time"]).sum(axis=1, skipna=True)
    y = pd.Series(y_series.values, index=raw["time"]).asfreq("30min")

    # Train‑test split: last 6 days = 288 half‑hours
    TEST_SIZE = 6 * 48
    y_train = y[:-TEST_SIZE]
    y_test = y[-TEST_SIZE:]

    # Impute missing values (forward‑fill, then back‑fill edge cases)
    y_train_filled = y_train.ffill().bfill()
    y_test_filled = y_test.ffill().bfill()

    # Z‑score scaling using training stats
    mu, std = y_train_filled.mean(), y_train_filled.std()
    y_train_scaled = ((y_train_filled - mu) / std).values.astype(np.float32)
    y_test_scaled = ((y_test_filled - mu) / std).values.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Prepare windowed datasets
    # ------------------------------------------------------------------
    lookback, horizon = args.lookback, args.horizon
    full_scaled = np.concatenate([y_train_scaled, y_test_scaled])
    train_ds = WindowDataset(y_train_scaled, lookback, horizon)

    # Hold out last 10% of training for validation
    n_train = int(len(train_ds) * 0.9)
    n_val = len(train_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = LSTMForecaster(n_feats=1, hidden=args.hidden, num_layers=args.layers,
                           horizon=horizon, dropout=args.dropout).to(device)
    crit = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Train with early stopping
    # ------------------------------------------------------------------
    best_val = float("inf")
    patience = args.patience
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, crit, optim, device)
        val_loss = eval_epoch(model, val_loader, crit, device)
        print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            patience = args.patience  # reset
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break
    model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 5. Rolling‑origin evaluation
    # ------------------------------------------------------------------
    results_dir = Path(f"results-lstm-L{lookback}-H{horizon}")
    plots_dir = results_dir / "plots"
    data_dir = results_dir / "data"
    results_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    all_metrics: Dict[int, Dict[str, float]] = {}
    all_predictions = {}
    
    for step in args.step_sizes:
        print(f"\nRolling predictions with step={step} …")
        preds = make_rolling_predictions(model, y_test_scaled.copy(), lookback,
                                         step, horizon, (mu, std), device)
        metrics = compute_metrics(y_test.values, preds)
        all_metrics[step] = metrics
        all_predictions[step] = pd.Series(preds, index=y_test.index)
        print(f"MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  MAPE={metrics['MAPE']:.2f}%")
        # Save predictions
        all_predictions[step].to_csv(data_dir / f"predictions_{step}step.csv")
        
        # Generate daily predictions plot for this step size
        plot_daily_predictions(y_test, all_predictions[step], step, plots_dir)

    # Generate aggregated plots
    plot_aggregated_predictions(y_test, all_predictions, plots_dir)
    plot_metrics_comparison(args.step_sizes, all_metrics, plots_dir)

    # Save metrics summary
    pd.DataFrame(all_metrics).T.to_csv(results_dir / "prediction_metrics.csv", index_label="Steps")

    # Save model
    torch.save({"state_dict": model.state_dict(), "mu": mu, "std": std}, results_dir / "model.pt")
    print(f"\n✔ Done – results in {results_dir.resolve()}")

if __name__ == "__main__":
    main()
