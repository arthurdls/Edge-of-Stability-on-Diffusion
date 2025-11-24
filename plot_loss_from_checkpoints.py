"""
Script to plot the loss from checkpoints in a directory.

Usage:
    python plot_loss_from_checkpoints.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Optional
import csv


def find_checkpoint_files(checkpoint_dir: Path) -> List[Tuple[Path, int]]:
    """
    Find all checkpoint files in the directory and extract epoch numbers.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        List of (checkpoint_path, epoch_number) tuples, sorted by epoch
    """
    checkpoint_files = []
    pattern = re.compile(r'checkpoint_epoch_(\d+)\.pt')
    
    for file_path in checkpoint_dir.glob('checkpoint_epoch_*.pt'):
        match = pattern.match(file_path.name)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((file_path, epoch))
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[1])
    return checkpoint_files


def extract_loss_from_checkpoints(
    checkpoint_dir: str,
    verbose: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Extract loss values from all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Path to directory containing checkpoints
        verbose: Print progress
        
    Returns:
        Tuple of (epochs, losses) lists
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if verbose:
        print(f"Finding checkpoints in: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    if verbose:
        print(f"Found {len(checkpoint_files)} checkpoint files")
    
    epochs = []
    losses = []
    
    # Process each checkpoint
    for checkpoint_path, epoch in checkpoint_files:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract loss
            if 'avg_epoch_loss' in checkpoint:
                loss = checkpoint['avg_epoch_loss']
                # Convert to Python float if it's a tensor
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                
                epochs.append(epoch)
                losses.append(loss)
                
                if verbose:
                    print(f"Epoch {epoch}: Loss = {loss:.6f}")
            else:
                if verbose:
                    print(f"Warning: No 'avg_epoch_loss' found in {checkpoint_path.name}")
                    
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path.name}: {e}")
            continue
    
    return epochs, losses


def plot_loss_history(
    epochs: List[int],
    losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Loss vs Epoch"
):
    """
    Plot loss values over epochs.
    
    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if len(epochs) == 0:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, losses, 'o-', linewidth=2, markersize=5, 
            label='Average Epoch Loss', color='blue', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def save_loss_to_csv(
    epochs: List[int],
    losses: List[float],
    csv_path: str
):
    """
    Save loss data to CSV file.
    
    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        csv_path: Path to save CSV file
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'avg_epoch_loss'])
        for epoch, loss in zip(epochs, losses):
            writer.writerow([epoch, loss])
    print(f"Loss data saved to: {csv_path}")


def main():
    """Main function to run the script."""
    # Configuration
    checkpoint_dir = "tests/test_adaptive_sampling_lr_1e-3"
    
    print("=" * 60)
    print("Plotting Loss from Checkpoints")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 60)
    
    # Extract loss from checkpoints
    epochs, losses = extract_loss_from_checkpoints(
        checkpoint_dir=checkpoint_dir,
        verbose=True
    )
    
    if len(epochs) == 0:
        print("No loss data found. Exiting.")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully extracted loss for {len(epochs)} checkpoints")
    print(f"Epoch range: {min(epochs)} - {max(epochs)}")
    print(f"Loss range: {min(losses):.6f} - {max(losses):.6f}")
    print(f"Mean loss: {np.mean(losses):.6f}")
    print(f"Std loss: {np.std(losses):.6f}")
    
    # Save results to CSV
    results_path = Path(checkpoint_dir) / "loss_results.csv"
    save_loss_to_csv(epochs, losses, str(results_path))
    
    # Plot results
    plot_path = Path(checkpoint_dir) / "loss_plot.png"
    plot_loss_history(
        epochs=epochs,
        losses=losses,
        save_path=str(plot_path),
        title="Training Loss vs Epoch"
    )


if __name__ == "__main__":
    main()

