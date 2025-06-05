import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_predictions(data_dir: Path):
    """Load all prediction files and test data."""
    # Load test data
    test_data = pd.read_csv('results-101111/data/test_data.csv', index_col=0, parse_dates=True).squeeze()

    # Load predictions for each step size
    predictions = {}
    step_sizes = [1, 3, 6, 12, 24]
    for n in step_sizes:
        pred_file = data_dir / f'predictions_{n}step.csv'
        predictions[n] = pd.read_csv(pred_file, index_col=0, parse_dates=True).squeeze()

    return test_data, predictions

def plot_last_day(test_data: pd.Series, predictions: dict, output_dir: Path):
    """Plot predictions for the last day with clean time-only x-axis."""
    # Get the last day's data
    last_day = test_data.index[-48:]  # Last 48 points (24 hours of 30-min data)

    plt.figure(figsize=(10, 6))

    # Plot actual data
    plt.plot(test_data[last_day].index.strftime('%H:%M'),
             test_data[last_day].values,
             label='Actual', linewidth=2.5, color='black')

    # Plot predictions for each step size
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (n_steps, pred), color in zip(predictions.items(), colors):
        plt.plot(pred[last_day].index.strftime('%H:%M'),
                 pred[last_day].values,
                 '--', label=f'{n_steps}-step', linewidth=1.5, alpha=0.8, color=color)

    # Customize the plot
    plt.title(f'Predictions Comparison for Day 40',
              fontsize=20, pad=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Usage', fontsize=16)

    # Set legend inside the plot with larger font
    plt.legend(fontsize=14, loc='lower right')

    # Set grid
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks every 2 hours (4 points since we have 30-min data)
    plt.xticks(test_data[last_day].index[::4].strftime('%H:%M'), fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_dir / 'last_day_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up directories
    data_dir = Path('results-lstm-L48-H24/data')
    output_dir = Path('results-lstm-L48-H24/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_data, predictions = load_predictions(data_dir)

    # Create plot
    plot_last_day(test_data, predictions, output_dir)
    print(f"Plot saved to {output_dir / 'last_day_predictions.png'}")

if __name__ == "__main__":
    main()
