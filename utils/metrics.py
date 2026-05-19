"""
Evaluation metrics and visualization utilities for stock prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def compute_prediction_metrics(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> dict[str, float]:
    """
    Compute comprehensive prediction metrics.

    Args:
        predictions: Predicted values
        targets: Ground truth values (same shape)

    Returns:
        Dictionary of metrics: MAE, MSE, RMSE, MAPE, R², Directional Accuracy
    """
    if predictions.ndim > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

    mae = float(np.mean(np.abs(predictions - targets)))
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))

    metrics: dict[str, float] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

    # MAPE (avoid division by zero)
    nonzero = np.abs(targets) > 1e-10
    if nonzero.any():
        metrics["MAPE"] = float(
            np.mean(np.abs((predictions[nonzero] - targets[nonzero]) / targets[nonzero])) * 100
        )

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics["R2"] = float(1 - ss_res / (ss_tot + 1e-10))

    # Directional Accuracy
    if predictions.shape[-1] >= 1:
        p = predictions[..., 0]
        t = targets[..., 0]
        pred_dir = np.sign(np.diff(p, axis=0))
        true_dir = np.sign(np.diff(t, axis=0))
        metrics["Directional_Accuracy"] = float(np.mean(pred_dir == true_dir) * 100)

    return metrics


def calculate_sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - (risk_free_rate / periods_per_year)
    if np.std(returns) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / np.std(returns))


def plot_predictions(
    historical: NDArray[np.float64],
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64] | None = None,
    upper_bound: NDArray[np.float64] | None = None,
    lower_bound: NDArray[np.float64] | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Stock Price Prediction",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot historical data and predictions."""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(historical.shape[1])]

    n_features = min(historical.shape[1], 4)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    if n_features == 1:
        axes = [axes]

    hist_idx = np.arange(len(historical))
    pred_idx = np.arange(len(historical), len(historical) + len(predictions))

    for i in range(n_features):
        ax = axes[i]
        ax.plot(hist_idx, historical[:, i], label="Historical", color="blue", lw=1.5)
        ax.plot(pred_idx, predictions[:, i], label="Prediction", color="red", lw=1.5)

        if targets is not None:
            ax.plot(pred_idx, targets[:, i], label="Actual", color="green",
                    ls="--", lw=1.5, alpha=0.7)

        if upper_bound is not None and lower_bound is not None:
            ax.fill_between(pred_idx, lower_bound[:, i], upper_bound[:, i],
                            color="red", alpha=0.15, label="95% CI")

        ax.set_ylabel(feature_names[i])
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=len(historical) - 1, color="gray", ls=":", alpha=0.7)

    axes[-1].set_xlabel("Time Steps")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    learning_rates: Sequence[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot training and validation loss curves."""
    n_plots = 2 if learning_rates is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
    axes_list = [axes] if n_plots == 1 else list(axes)

    epochs = range(1, len(train_losses) + 1)

    ax = axes_list[0]
    ax.plot(epochs, train_losses, label="Training Loss", color="blue", lw=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss", color="red", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if learning_rates is not None and len(axes_list) > 1:
        ax = axes_list[1]
        ax.plot(epochs, learning_rates, label="Learning Rate", color="green", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
