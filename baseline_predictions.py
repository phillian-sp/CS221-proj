"""
baseline_predictions.py
=====================
Implementation of a baseline prediction model that calculates daily averages of the first 34 days
and uses them to predict the last 6 days.

Usage: python baseline_predictions.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def load_data() -> pd.Series:
    """Load and prepare the time series data."""
    CSV_PATH = "data/all_usage_halfhour.csv"
    raw = pd.read_csv(CSV_PATH, parse_dates=["time"])

    # Sum every numeric company column for each 30-minute stamp
    total_usage = raw.drop(columns=["time"]).sum(axis=1, skipna=True)

    # Build half-hourly series with frequency
    y = pd.Series(total_usage.values, index=raw["time"]).asfreq("30min")
    return y[:-3]  # remove last three rows (NaN values)


def split_data(data: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Split data into training (first 34 days) and test (last 6 days)."""
    test_size = 6 * 48  # 6 days of 30-minute data
    train = data[:-test_size]
    test = data[-test_size:]
    return train, test


def calculate_daily_averages(train_data: pd.Series) -> pd.Series:
    """
    Calculate average values for each 30-minute interval of the day,
    using the training data (first 34 days).
    """
    # Extract time of day and create a grouping key (0 to 47 for each 30-min interval)
    train_data = train_data.copy()
    train_data.index = pd.DatetimeIndex(train_data.index)
    time_of_day = (train_data.index.hour * 2 + train_data.index.minute // 30) % 48
    
    # Group by time of day and calculate mean
    daily_avg = train_data.groupby(time_of_day).mean()
    return daily_avg


def generate_predictions(test_data: pd.Series, daily_averages: pd.Series) -> pd.Series:
    """
    Generate predictions for test data using daily averages.
    For each timestamp in test data, use the average value for that time of day.
    """
    test_index = test_data.index
    time_of_day = (test_index.hour * 2 + test_index.minute // 30) % 48
    
    # Create predictions by mapping each test timestamp to its corresponding daily average
    predictions = pd.Series(
        index=test_index,
        data=[daily_averages[tod] for tod in time_of_day]
    )
    return predictions


def calculate_metrics(test_data: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    """Calculate error metrics for predictions."""
    errors = predictions - test_data
    metrics = {
        'MAE': np.mean(np.abs(errors)),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'MAPE': np.mean(np.abs(errors / test_data)) * 100
    }
    return metrics


def plot_daily_predictions(test_data: pd.Series, predictions: pd.Series, plots_dir: Path):
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
                label='Baseline Predictions', linewidth=2)

        ax.set_title(f'Day {day + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Daily Baseline Predictions (Daily Average Model)', y=1.02)
    plt.tight_layout()

    # Save plot
    plt.savefig(plots_dir / 'baseline_daily_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_day_six(test_data: pd.Series, predictions: pd.Series, plots_dir: Path):
    """Plot predictions vs actual for only day 6 with improved formatting."""
    # Get the last day's data (day 6)
    days = pd.Timedelta(days=1)
    start_time = test_data.index[0]
    day_start = start_time + 5 * days  # Day 6 is index 5 (0-based)
    day_end = day_start + days
    
    day_six_data = test_data[day_start:day_end]
    day_six_pred = predictions[day_start:day_end]
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.plot(day_six_data.index.strftime('%H:%M'),
             day_six_data.values,
             label='Actual', linewidth=2.5, color='black')
    
    # Plot baseline prediction
    plt.plot(day_six_pred.index.strftime('%H:%M'),
             day_six_pred.values,
             '--', label='Baseline Prediction', linewidth=1.5, alpha=0.8, color='#1f77b4')
    
    # Customize the plot
    plt.title(f'Baseline Model Predictions for Day 40',
              fontsize=20, pad=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Usage', fontsize=16)
    
    # Set legend inside the plot with larger font
    plt.legend(fontsize=14, loc='lower right')
    
    # Set grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks every 2 hours (4 points since we have 30-min data)
    plt.xticks(day_six_data.index[::4].strftime('%H:%M'), fontsize=14)
    plt.yticks(fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(plots_dir / 'day_six_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Create results directory
    RESULTS_DIR = Path("results-baseline")
    PLOTS_DIR = RESULTS_DIR / "plots"
    DATA_DIR = RESULTS_DIR / "data"

    # Create directories
    for dir_path in [RESULTS_DIR, PLOTS_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Starting baseline analysis...")

    # 1. Load Data
    print("Loading data...")
    data = load_data()

    # 2. Train/Test Split (last 6 days as test)
    train, test = split_data(data)
    
    print("Data shapes:")
    print(f"Training data: {len(train)} observations")
    print(f"Test data: {len(test)} observations")

    # Save train/test split
    train.to_csv(DATA_DIR / 'train_data.csv')
    test.to_csv(DATA_DIR / 'test_data.csv')

    # 3. Calculate daily averages from training data
    print("Calculating daily averages...")
    daily_averages = calculate_daily_averages(train)
    daily_averages.to_csv(DATA_DIR / 'daily_averages.csv')
    
    # 4. Generate predictions for test data
    print("Generating predictions...")
    predictions = generate_predictions(test, daily_averages)
    predictions.to_csv(DATA_DIR / 'baseline_predictions.csv')
    
    # 5. Calculate metrics
    metrics = calculate_metrics(test, predictions)
    print("\nBaseline Model Metrics:")
    print(f"MAE:  {metrics['MAE']:.3f}")
    print(f"RMSE: {metrics['RMSE']:.3f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / 'baseline_metrics.csv', index=False)
    
    # 6. Create visualizations
    print("Creating visualizations...")
    plot_daily_predictions(test, predictions, PLOTS_DIR)
    
    # 7. Create day 6 specific plot
    print("Creating day 6 specific plot...")
    plot_day_six(test, predictions, PLOTS_DIR)
    
    print(f"\nBaseline analysis complete! Results saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    main() 