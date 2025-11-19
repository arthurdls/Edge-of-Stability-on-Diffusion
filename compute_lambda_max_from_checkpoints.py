"""
Script to compute lambda max (largest Hessian eigenvalue) from multiple checkpoints
and plot the results over training epochs.

Usage:
    python compute_lambda_max_from_checkpoints.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Tuple, Optional

# Import from the implementations
from implementations.base_implementation import UNet, DiffusionSchedule
from implementations.utils.measure_checkpoint import (
    compute_lambda_max_from_checkpoint_simple,
    create_preconditioner_from_checkpoint,
    create_constant_preconditioner
)


def infer_model_architecture(checkpoint_path: Path) -> dict:
    """
    Infer model architecture parameters from a checkpoint file.
    
    Args:
        checkpoint_path: Path to a checkpoint file
        
    Returns:
        dict with keys: in_ch, base_ch, time_emb_dim
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('model_state', checkpoint)
    
    # Infer from conv_in layer
    conv_in_shape = state['conv_in.weight'].shape
    in_ch = conv_in_shape[1]
    base_ch = conv_in_shape[0]
    
    # Infer time_emb_dim from time_mlp first layer
    time_mlp_shape = state['time_mlp.0.weight'].shape
    time_emb_dim = time_mlp_shape[1]
    
    return {
        'in_ch': in_ch,
        'base_ch': base_ch,
        'time_emb_dim': time_emb_dim
    }


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


def compute_lambda_max_for_checkpoints(
    checkpoint_dir: str,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    image_size: Tuple[int, int, int] = (3, 32, 32),  # CIFAR-10: 3 channels, 32x32
    timesteps: int = 50,
    max_iterations: int = 100,
    reltol: float = 1e-2,
    use_power_iteration: bool = False,
    use_ema: bool = False,
    verbose: bool = True,
    use_preconditioner: bool = False,
    optimizer_type: str = 'auto',
    learning_rate: Optional[float] = None
) -> Tuple[List[int], List[float]]:
    """
    Compute lambda max for all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Path to directory containing checkpoints
        device: Device to run computation on (default: cuda if available, else cpu)
        batch_size: Batch size for computing loss
        image_size: Image dimensions (channels, height, width)
        timesteps: Number of diffusion timesteps
        max_iterations: Maximum iterations for eigenvalue computation
        reltol: Relative tolerance for convergence
        use_power_iteration: Use power iteration instead of LOBPCG
        use_ema: Load EMA weights instead of regular weights
        verbose: Print progress
        use_preconditioner: If True, compute preconditioned lambda max (P^{-1} H)
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'sgd', or 'auto' to infer)
        learning_rate: Learning rate for constant preconditioner (required if optimizer state not available)
        
    Returns:
        Tuple of (epochs, lambda_max_values) lists
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Finding checkpoints in: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    if verbose:
        print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Infer model architecture from first checkpoint
    first_checkpoint_path, _ = checkpoint_files[0]
    arch_params = infer_model_architecture(first_checkpoint_path)
    if verbose:
        print(f"Inferred architecture from checkpoint: {arch_params}")
    
    # Initialize model and schedule (architecture only, weights will be loaded from checkpoint)
    # Use inferred architecture to match the checkpoint
    model = UNet(
        in_ch=arch_params['in_ch'],
        base_ch=arch_params['base_ch'],
        time_emb_dim=arch_params['time_emb_dim'],
        max_t=1000.0
    )
    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    
    # Create a sample data batch for computing loss
    # This will be used for all checkpoints
    x_sample = torch.randn(batch_size, *image_size, device=device)
    
    # Store results
    epochs = []
    lambda_max_values = []
    
    # Process each checkpoint
    for checkpoint_path, epoch in tqdm(checkpoint_files, desc="Computing lambda max"):
        try:
            if verbose:
                print(f"\nProcessing checkpoint: {checkpoint_path.name} (epoch {epoch})")
            
            # Extract or create preconditioner if requested
            P = None
            if use_preconditioner:
                if verbose:
                    print(f"  Extracting preconditioner from checkpoint...")
                P = create_preconditioner_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer_type=optimizer_type,
                    lr=learning_rate,
                    device=device
                )
                
                # Fallback to constant preconditioner if optimizer state not available
                if P is None:
                    if learning_rate is None:
                        if verbose:
                            print(f"  ⚠️  Warning: No optimizer state found in checkpoint and no learning rate provided.")
                            print(f"     Skipping preconditioner. Computing regular lambda max.")
                            print(f"     To compute preconditioned lambda max, set 'learning_rate' in main() function")
                            print(f"     (e.g., learning_rate=1e-2 for SGD or learning_rate=2e-4 for Adam)")
                    else:
                        if verbose:
                            print(f"  No optimizer state found. Creating constant preconditioner with lr={learning_rate}")
                        P = create_constant_preconditioner(model, lr=learning_rate, device=device)
            
            # Compute lambda max for this checkpoint
            lambda_max = compute_lambda_max_from_checkpoint_simple(
                checkpoint_path=checkpoint_path,
                model=model,
                schedule=schedule,
                x_sample=x_sample,
                t_sample=None,  # Will generate random timesteps
                device=device,
                use_ema=use_ema,
                k=1,  # Only compute largest eigenvalue
                max_iterations=max_iterations,
                reltol=reltol,
                use_power_iteration=use_power_iteration,
                return_eigenvectors=False,
                P=P  # Pass preconditioner if available
            )
            
            # Convert to Python float if it's a tensor
            if isinstance(lambda_max, torch.Tensor):
                lambda_max = lambda_max.item()
            
            epochs.append(epoch)
            lambda_max_values.append(lambda_max)
            
            if verbose:
                print(f"  Lambda max: {lambda_max:.6f}")
                
        except Exception as e:
            print(f"  Error processing checkpoint {checkpoint_path.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    return epochs, lambda_max_values


def plot_lambda_max_history(
    epochs: List[int],
    lambda_max_values: List[float],
    save_path: Optional[str] = None,
    title: str = "Lambda Max vs Epoch",
    is_preconditioned: bool = False
):
    """
    Plot lambda max values over epochs.
    
    Args:
        epochs: List of epoch numbers
        lambda_max_values: List of lambda max values
        save_path: Path to save the plot (optional)
        title: Plot title
        is_preconditioned: If True, indicates this is preconditioned lambda max
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lambda_max_values, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    if is_preconditioned:
        plt.ylabel('Lambda Max (Largest Eigenvalue of P^{-1} H)', fontsize=12)
    else:
        plt.ylabel('Lambda Max (Largest Hessian Eigenvalue)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the script."""
    # Configuration
    # Try multiple possible checkpoint directories
    possible_dirs = ["runs_ddim", "tests/runs_ddim", "../runs_ddim"]
    checkpoint_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists() and len(list(Path(dir_path).glob("checkpoint_epoch_*.pt"))) > 0:
            checkpoint_dir = dir_path
            break
    
    if checkpoint_dir is None:
        checkpoint_dir = "runs_ddim"  # Default fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters for lambda max computation
    batch_size = 4
    image_size = (3, 32, 32)  # CIFAR-10
    timesteps = 50
    max_iterations = 100
    reltol = 1e-2
    use_power_iteration = False  # Use LOBPCG by default (faster)
    use_ema = False  # Set to True if you want to use EMA weights
    
    # Preconditioner settings
    use_preconditioner = True  # Set to True to compute preconditioned lambda max (P^{-1} H)
    optimizer_type = 'auto'  # 'auto', 'adam', 'rmsprop', or 'sgd'
    # Learning rate for constant preconditioner (required if optimizer state not available)
    # Set this to the learning rate you used during training:
    # - For SGD (edm_style_preconditioning.py): typically 1e-2
    # - For Adam (base_implementation.py): typically 2e-4
    # If your checkpoints have optimizer state, this will be ignored
    learning_rate = 1e-2  # TODO: Change this to match your training learning rate!
    
    print("=" * 60)
    print("Computing Lambda Max from Checkpoints")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Timesteps: {timesteps}")
    print(f"Max iterations: {max_iterations}")
    print(f"Relative tolerance: {reltol}")
    print(f"Use power iteration: {use_power_iteration}")
    print(f"Use EMA weights: {use_ema}")
    print(f"Use preconditioner: {use_preconditioner}")
    if use_preconditioner:
        print(f"Optimizer type: {optimizer_type}")
        print(f"Learning rate: {learning_rate if learning_rate else 'auto (from checkpoint)'}")
    print("=" * 60)
    
    # Compute lambda max for all checkpoints
    epochs, lambda_max_values = compute_lambda_max_for_checkpoints(
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        timesteps=timesteps,
        max_iterations=max_iterations,
        reltol=reltol,
        use_power_iteration=use_power_iteration,
        use_ema=use_ema,
        verbose=True,
        use_preconditioner=use_preconditioner,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate
    )
    
    if len(epochs) == 0:
        print("No successful computations. Exiting.")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully computed lambda max for {len(epochs)} checkpoints")
    print(f"Epoch range: {min(epochs)} - {max(epochs)}")
    print(f"Lambda max range: {min(lambda_max_values):.6f} - {max(lambda_max_values):.6f}")
    print(f"Mean lambda max: {np.mean(lambda_max_values):.6f}")
    print(f"Std lambda max: {np.std(lambda_max_values):.6f}")
    
    # Save results to CSV
    suffix = "_preconditioned" if use_preconditioner else ""
    results_path = Path(checkpoint_dir) / f"lambda_max_results{suffix}.csv"
    import csv
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lambda_max'])
        for epoch, lambda_max in zip(epochs, lambda_max_values):
            writer.writerow([epoch, lambda_max])
    print(f"\nResults saved to: {results_path}")
    
    # Plot results
    plot_path = Path(checkpoint_dir) / f"lambda_max_plot{suffix}.png"
    title_suffix = " (Preconditioned P^{-1} H)" if use_preconditioner else ""
    plot_lambda_max_history(
        epochs=epochs,
        lambda_max_values=lambda_max_values,
        save_path=str(plot_path),
        title=f"Lambda Max (Largest Hessian Eigenvalue){title_suffix} vs Epoch",
        is_preconditioned=use_preconditioner
    )


if __name__ == "__main__":
    main()

