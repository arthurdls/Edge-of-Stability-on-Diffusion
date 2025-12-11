"""
Visualization utilities for lambda max (Hessian eigenvalue) tracking.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


def plot_lambda_max(
    lambda_max_history: List[Dict],
    save_path: Optional[str] = None,
    figsize=(12, 6),
    show_loss=True,
    learning_rate: Optional[float] = None
):
    """
    Plot lambda max (largest Hessian eigenvalue) over training steps.
    
    Args:
        lambda_max_history: List of dictionaries with 'step', 'lambda_max', and optionally 'loss'
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show_loss: If True, plot loss on secondary y-axis
        learning_rate: Optional learning rate to plot 2/eta threshold line
    """
    if not lambda_max_history:
        print("No lambda max history to plot")
        return
    
    steps = [entry['step'] for entry in lambda_max_history]
    lambda_max_values = [entry['lambda_max'] for entry in lambda_max_history]
    losses = [entry.get('loss', None) for entry in lambda_max_history]
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot lambda max
    ax1.plot(steps, lambda_max_values, 'b-', linewidth=2, label=r'$\lambda_{max}$ (Hessian)', marker='o', markersize=3)
    
    # Plot 2/eta threshold if learning rate is provided
    if learning_rate is not None:
        threshold = 2.0 / learning_rate
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=r'$2/\eta$ threshold')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel(r'$\lambda_{max}$ (Largest Hessian Eigenvalue)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot loss on secondary axis if requested
    if show_loss and any(l is not None for l in losses):
        ax2 = ax1.twinx()
        valid_losses = [(s, l) for s, l in zip(steps, losses) if l is not None]
        if valid_losses:
            loss_steps, loss_values = zip(*valid_losses)
            ax2.plot(loss_steps, loss_values, 'g--', alpha=0.6, linewidth=1.5, label='Loss')
            ax2.set_ylabel('Loss', fontsize=12, color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_yscale('log')
            ax2.legend(loc='upper right')
    
    plt.title('Hessian Largest Eigenvalue ($\lambda_{max}$) During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_lambda_max_multiple(
    histories_dict: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
    figsize=(12, 6),
    learning_rate: Optional[float] = None
):
    """
    Plot multiple lambda max histories (e.g., from different runs) on the same plot.
    
    Args:
        histories_dict: Dictionary mapping run names to lambda_max_history lists
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        learning_rate: Optional learning rate to plot 2/eta threshold line
    """
    if not histories_dict:
        print("No histories to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    for (run_name, history), color in zip(histories_dict.items(), colors):
        if not history:
            continue
        steps = [entry['step'] for entry in history]
        lambda_max_values = [entry['lambda_max'] for entry in history]
        ax.plot(steps, lambda_max_values, color=color, linewidth=2, label=run_name, marker='o', markersize=3)
    
    # Plot 2/eta threshold if learning rate is provided
    if learning_rate is not None:
        threshold = 2.0 / learning_rate
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=r'$2/\eta$ threshold')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(r'$\lambda_{max}$ (Largest Hessian Eigenvalue)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.title('Hessian Largest Eigenvalue ($\lambda_{max}$) Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def save_lambda_max_history(
    lambda_max_history: List[Dict],
    save_path: str
):
    """
    Save lambda max history to a CSV file.
    
    Args:
        lambda_max_history: List of dictionaries with lambda max data
        save_path: Path to save CSV file
    """
    import csv
    
    if not lambda_max_history:
        print("No lambda max history to save")
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='') as f:
        fieldnames = ['step', 'epoch', 'lambda_max', 'loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in lambda_max_history:
            writer.writerow({
                'step': entry['step'],
                'epoch': entry.get('epoch', ''),
                'lambda_max': entry['lambda_max'],
                'loss': entry.get('loss', '')
            })
    
    print(f"Saved lambda max history to {save_path}")


def load_lambda_max_history(load_path: str) -> List[Dict]:
    """
    Load lambda max history from a CSV file.
    
    Args:
        load_path: Path to CSV file
        
    Returns:
        List of dictionaries with lambda max data
    """
    import csv
    
    history = []
    with open(load_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                'step': int(row['step']),
                'epoch': int(row['epoch']) if row['epoch'] else None,
                'lambda_max': float(row['lambda_max']),
                'loss': float(row['loss']) if row['loss'] else None
            })
    
    return history


