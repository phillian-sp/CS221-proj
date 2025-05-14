"""
rolling_predictions.py
=====================
Analysis of rolling predictions with different step sizes using SARIMAX model.
Usage: python rolling_predictions.py <model_order>
Example: python rolling_predictions.py --model-order 101111
    -> This will use SARIMAX(1,0,1)x(1,1,1,48)
Example: python rolling_predictions.py --model-order 101111 --step-sizes 1 3 6 12 24
    -> This will use SARIMAX(1,0,1)x(1,1,1,48) and perform rolling predictions with step sizes 1, 3, 6, 12, and 24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path
import argparse

def parse_model_order(order_str: str) -> tuple:
    """Parse model order string into SARIMAX parameters."""
    if len(order_str) != 6:
        raise ValueError("Model order must be 6 digits (e.g., '101111')")

    try:
        # Parse into individual integers
        p, d, q, P, D, Q = map(int, order_str)
        # Return regular and seasonal orders
        return (p, d, q), (P, D, Q, 48)
    except ValueError:
        raise ValueError("Model order must contain only digits")

def get_results_dir(order_str: str) -> Path:
    """Create results directory name based on model order."""
    return Path(f"results-{order_str}")

def load_data() -> pd.Series:
    """Load and prepare the time series data."""
    CSV_PATH = "data/all_usage_halfhour.csv"
    raw = pd.read_csv(CSV_PATH, parse_dates=["time"])

    # Sum every numeric company column for each 30-minute stamp
    total_usage = raw.drop(columns=["time"]).sum(axis=1, skipna=True)

    # Build half-hourly series with frequency
    y = pd.Series(total_usage.values, index=raw["time"]).asfreq("30min")
    return y[:-3]  # remove last three rows (NaN values)

def rolling_predict(model_fit, test_data: pd.Series, n_steps: int) -> Tuple[pd.Series, List[float]]:
    """
    Perform rolling predictions with n-step updates.

    Args:
        model_fit: Fitted SARIMAX model
        test_data: Test data series
        n_steps: Number of steps to predict before updating state

    Returns:
        Tuple of (predictions Series, computation times list)
    """
    predictions = []
    pred_index = []
    computation_times = []

    # Calculate number of blocks needed
    n_blocks = int(np.ceil(len(test_data) / n_steps))

    # Use tqdm for progress tracking
    for i in tqdm(range(n_blocks), desc=f"Making {n_steps}-step predictions"):
        # Start time for this prediction block
        start_idx = i * n_steps
        end_idx = min(start_idx + n_steps, len(test_data))

        # Make n-step prediction
        start_time = pd.Timestamp.now()
        forecast = model_fit.forecast(steps=n_steps)
        computation_times.append((pd.Timestamp.now() - start_time).total_seconds())

        # Store only the predictions we need
        predictions.extend(forecast[:end_idx - start_idx])
        pred_index.extend(test_data.index[start_idx:end_idx])

        # Update model state with actual values if not at the end
        if end_idx < len(test_data):
            actual_values = test_data.iloc[start_idx:end_idx]
            model_fit = model_fit.append(actual_values)

    return pd.Series(predictions, index=pred_index), computation_times

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

    plt.suptitle(f'Daily Predictions with {n_steps}-step Updates', y=1.02)
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

    plt.suptitle('Daily Predictions Comparison', y=1.02)
    plt.tight_layout()

    # Save plot
    plt.savefig(plots_dir / 'aggregated_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(test_data: pd.Series, predictions: pd.Series) -> dict:
    """Calculate error metrics for predictions."""
    errors = predictions - test_data
    metrics = {
        'MAE': np.mean(np.abs(errors)),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'MAPE': np.mean(np.abs(errors / test_data)) * 100
    }
    return metrics

def create_model_copy(original_model, params):
    """Create a new model with the same parameters as the original."""
    model_copy = SARIMAX(
        original_model.data.endog,
        order=original_model.model.order,
        seasonal_order=original_model.model.seasonal_order,
        enforce_stationarity=original_model.model.enforce_stationarity,
        enforce_invertibility=original_model.model.enforce_invertibility,
        measurement_error=original_model.model.measurement_error,
        time_varying_regression=original_model.model.time_varying_regression,
        mle_regression=original_model.model.mle_regression,
        simple_differencing=original_model.model.simple_differencing,
        trend=original_model.model.trend
    )
    return model_copy.filter(params)

def plot_metrics_comparison(step_sizes: List[int], all_metrics: dict, plots_dir: Path):
    """Plot and save metrics comparison."""
    plt.figure(figsize=(12, 6))
    for metric in ['MAE', 'RMSE', 'MAPE']:
        plt.plot(step_sizes, [all_metrics[n][metric] for n in step_sizes],
                 marker='o', label=metric)
    plt.xlabel('Prediction Steps')
    plt.ylabel('Error')
    plt.title('Error Metrics vs Prediction Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform rolling predictions using SARIMAX model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Example:
                python rolling_predictions.py --model-order 101111
                    -> This will use SARIMAX(1,0,1)x(1,1,1,48)

            Model Order Format:
                6 digits representing (p,d,q)(P,D,Q) parameters
                where:
                - p,d,q: non-seasonal order parameters
                - P,D,Q: seasonal order parameters
                - Seasonal period is fixed at 48 (daily seasonality for 30-min data)
            ''')
    )

    parser.add_argument(
        '--model-order',
        type=str,
        required=True,
        help='SARIMAX model order as 6 digits (e.g., 101111 for SARIMAX(1,0,1)x(1,1,1,48))'
    )

    parser.add_argument(
        '--step-sizes',
        type=int,
        nargs='+',
        default=[1, 3, 6, 12, 24],
        help='List of step sizes for rolling predictions (default: 1 3 6 12 24)'
    )

    return parser.parse_args()

def load_saved_results(results_dir: Path, data_dir: Path) -> tuple:
    """Load saved results from previous run."""
    print(f"Loading existing results from {results_dir}")

    # Load metrics and predictions
    metrics_df = pd.read_csv(results_dir / 'prediction_metrics.csv', index_col='Steps')
    times_df = pd.read_csv(results_dir / 'computation_times_summary.csv')

    # Load test data and predictions
    test = pd.read_csv(data_dir / 'test_data.csv', index_col=0, parse_dates=True).squeeze()

    # Load predictions for each step size
    all_predictions = {}
    step_sizes = times_df['step_size'].tolist()
    for n in step_sizes:
        pred_file = data_dir / f'predictions_{n}step.csv'
        if pred_file.exists():
            all_predictions[n] = pd.read_csv(pred_file, index_col=0, parse_dates=True).squeeze()

    return test, all_predictions, metrics_df.to_dict('index'), times_df, step_sizes

def generate_plots(test_data: pd.Series, all_predictions: dict, all_metrics: dict,
                  times_df: pd.DataFrame, step_sizes: List[int], plots_dir: Path):
    """Generate all plots from results."""
    print("\nGenerating plots...")

    # Create plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual step predictions plots
    for n in step_sizes:
        if n in all_predictions:
            plot_daily_predictions(test_data, all_predictions[n], n, plots_dir)

    # Generate aggregated plots
    plot_aggregated_predictions(test_data, all_predictions, plots_dir)

    # Generate metrics comparison plot
    plot_metrics_comparison(step_sizes, all_metrics, plots_dir)

    print(f"Plots saved to {plots_dir}")

def main():
    # Parse command line arguments
    args = parse_args()

    try:
        order, seasonal_order = parse_model_order(args.model_order)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create results directory with model order
    RESULTS_DIR = get_results_dir(args.model_order)
    PLOTS_DIR = RESULTS_DIR / "plots"
    DATA_DIR = RESULTS_DIR / "data"

    # Check if results already exist
    if RESULTS_DIR.exists():
        try:
            # Load existing results
            test, all_predictions, all_metrics, times_df, existing_steps = load_saved_results(RESULTS_DIR, DATA_DIR)

            # Check if we need to warn about different step sizes
            if set(args.step_sizes) != set(existing_steps):
                print("\nWarning: Requested step sizes differ from saved results.")
                print(f"Using existing step sizes: {existing_steps}")
                print("To use different step sizes, please delete the results directory first.")

            # Generate plots from saved results
            generate_plots(test, all_predictions, all_metrics, times_df, existing_steps, PLOTS_DIR)
            return 0

        except Exception as e:
            print(f"\nError loading existing results: {e}")
            print("Proceeding with new analysis...")

    # Create directories for new analysis
    for dir_path in [RESULTS_DIR, PLOTS_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Using SARIMAX{order}x{seasonal_order}")
    print(f"Results will be saved to {RESULTS_DIR}")

    # Save model configuration
    with open(RESULTS_DIR / "model_config.txt", "w") as f:
        f.write(f"Model: SARIMAX{order}x{seasonal_order}\n")
        f.write(f"Order string: {args.model_order}\n")
        f.write(f"Regular order (p,d,q): {order}\n")
        f.write(f"Seasonal order (P,D,Q,s): {seasonal_order}\n")
        f.write(f"Step sizes: {args.step_sizes}\n")

    print("Starting analysis...")

    # 1. Load Data
    print("Loading data...")
    y = load_data()

    # 2. Train/Test Split (last 6 days as test)
    test_size = 6 * 48  # 6 days of 30-minute data
    train = y[:-test_size]
    test = y[-test_size:]

    print("Data shapes:")
    print(f"Training data: {len(train)} observations")
    print(f"Test data: {len(test)} observations")

    # Save train/test split
    train.to_csv(DATA_DIR / 'train_data.csv')
    test.to_csv(DATA_DIR / 'test_data.csv')

    # 3. Fit Initial Model
    print("\nFitting initial model...")
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        measurement_error=True,
        trend='c'
    )
    initial_fit = model.fit(disp=False)
    print(initial_fit.summary())

    # Save model summary
    with open(RESULTS_DIR / 'model_summary.txt', 'w') as f:
        f.write(str(initial_fit.summary()))

    # 4. Perform Rolling Predictions with different n_steps
    all_predictions = {}
    all_metrics = {}
    all_times = {}

    for n in args.step_sizes:
        print(f"\nPerforming {n}-step predictions...")
        # Create a new model with the same parameters
        model_copy = create_model_copy(initial_fit, initial_fit.params)
        predictions, times = rolling_predict(model_copy, test, n)

        # Store results
        all_predictions[n] = predictions
        all_metrics[n] = calculate_metrics(test, predictions)
        all_times[n] = times

        # Save predictions
        predictions.to_csv(DATA_DIR / f'predictions_{n}step.csv')

        print(f"\nMetrics for {n}-step predictions:")
        print(f"MAE:  {all_metrics[n]['MAE']:.3f}")
        print(f"RMSE: {all_metrics[n]['RMSE']:.3f}")
        print(f"MAPE: {all_metrics[n]['MAPE']:.2f}%")
        print(f"Mean computation time per prediction: {np.mean(times):.3f}s")

    # 5. Save Summary Table
    print("\nSaving summary metrics...")
    summary_df = pd.DataFrame(all_metrics).T
    summary_df.index.name = 'Steps'
    summary_df.columns = ['MAE', 'RMSE', 'MAPE']
    summary_df.to_csv(RESULTS_DIR / 'prediction_metrics.csv')
    print("\nMetrics Summary:")
    print(summary_df.round(3))

    # Save computation times with proper structure
    times_summary = pd.DataFrame({
        'step_size': args.step_sizes,
        'mean_time': [np.mean(all_times[n]) for n in args.step_sizes],
        'min_time': [np.min(all_times[n]) for n in args.step_sizes],
        'max_time': [np.max(all_times[n]) for n in args.step_sizes],
        'total_time': [np.sum(all_times[n]) for n in args.step_sizes],
        'n_predictions': [len(all_times[n]) for n in args.step_sizes]
    })
    times_summary.to_csv(RESULTS_DIR / 'computation_times_summary.csv', index=False)

    # Save detailed times separately for each step size
    for n in args.step_sizes:
        pd.Series(all_times[n], name='computation_time').to_csv(
            DATA_DIR / f'computation_times_{n}step.csv'
        )

    # 6. Generate all plots
    generate_plots(test, all_predictions, all_metrics, times_summary, args.step_sizes, PLOTS_DIR)

    print(f"\nAnalysis complete! Results saved to {RESULTS_DIR}")
    print(f"Data saved to {DATA_DIR}")
    return 0

if __name__ == "__main__":
    import textwrap
    import sys
    sys.exit(main())
