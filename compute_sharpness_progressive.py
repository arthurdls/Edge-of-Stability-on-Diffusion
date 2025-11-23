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
from torchvision import datasets

# Import from the implementations
from implementations.base_implementation import UNet, DiffusionSchedule
from implementations.utils.measure_checkpoint_progressive import (
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
    
    # Try different possible checkpoint structures
    state = None
    if 'model_state' in checkpoint:
        state = checkpoint['model_state']
    elif 'ema_state' in checkpoint:
        state = checkpoint['ema_state']
    else:
        # Try using checkpoint directly
        state = checkpoint
    
    # Get all keys to help debug
    all_keys = list(state.keys()) if isinstance(state, dict) else []
    
    # Try to find conv_in layer with different possible key names
    conv_in_key = None
    for possible_key in ['conv_in.weight', 'model.conv_in.weight', 'net.conv_in.weight']:
        if possible_key in state:
            conv_in_key = possible_key
            break
    
    # If still not found, try to find any key containing 'conv_in'
    if conv_in_key is None:
        for key in all_keys:
            if 'conv_in' in key and 'weight' in key:
                conv_in_key = key
                break
    
    if conv_in_key is None:
        raise ValueError(
            f"Could not find conv_in layer in checkpoint. Available keys (first 20): {all_keys[:20]}\n"
            f"Checkpoint structure: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}"
        )
    
    conv_in_shape = state[conv_in_key].shape
    in_ch = conv_in_shape[1]
    base_ch = conv_in_shape[0]
    
    # Try to find time_mlp layer
    time_mlp_key = None
    for possible_key in ['time_mlp.0.weight', 'model.time_mlp.0.weight', 'net.time_mlp.0.weight']:
        if possible_key in state:
            time_mlp_key = possible_key
            break
    
    # If still not found, try to find any key containing 'time_mlp'
    if time_mlp_key is None:
        for key in all_keys:
            if 'time_mlp' in key and '0' in key and 'weight' in key:
                time_mlp_key = key
                break
    
    if time_mlp_key is None:
        # Default to base_ch if we can't find time_mlp
        time_emb_dim = base_ch
        print(f"Warning: Could not find time_mlp in checkpoint, using time_emb_dim={time_emb_dim} (same as base_ch)")
    else:
        time_mlp_shape = state[time_mlp_key].shape
        time_emb_dim = time_mlp_shape[1]
    
    return {
        'in_ch': in_ch,
        'base_ch': base_ch,
        'time_emb_dim': time_emb_dim
    }


def load_cifar10_sample(
    batch_size: int,
    image_size: Tuple[int, int] = (32, 32),
    data_root: str = './data',
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Load a batch of CIFAR-10 images for computing lambda max.
    
    Args:
        batch_size: Number of images to load
        image_size: Image size (height, width) - should be (32, 32) for CIFAR-10
        data_root: Root directory for CIFAR-10 dataset
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tensor of shape (batch_size, 3, height, width) with values in [-1, 1]
    """
    # Load CIFAR-10 dataset without transforms to avoid numpy issues
    # We'll manually convert to tensor and normalize
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=None  # Load raw data
    )
    
    # Set random seed if provided
    # Note: We don't set torch.use_deterministic_algorithms() here because
    # it causes issues with CuBLAS on CUDA and isn't needed for data loading
    if seed is not None:
        torch.manual_seed(seed)
        try:
            np.random.seed(seed)
        except (AttributeError, NameError):
            # numpy might not be available or compatible, skip it
            pass
    
    # Sample random indices
    indices = torch.randperm(len(dataset))[:batch_size]
    
    # Load the images and convert manually
    # CIFAR-10 returns PIL Images when transform=None, so we convert directly to tensor
    images = []
    for idx in indices:
        # Get PIL Image from dataset
        pil_img = dataset[idx][0]
        
        # Convert PIL Image to tensor manually to avoid numpy issues
        # PIL Image is in HWC format, we need CHW
        # Get pixel data as a list and reshape
        img_data = list(pil_img.getdata())
        img_width, img_height = pil_img.size
        channels = len(pil_img.getbands())
        
        # Reshape to (height, width, channels) and convert to tensor
        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        img_tensor = img_tensor.reshape(img_height, img_width, channels)
        
        # Convert from HWC to CHW
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # Normalize from [0, 255] to [-1, 1]
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
        
        images.append(img_tensor)
    
    # Stack into a batch tensor
    x_sample = torch.stack(images)
    
    return x_sample


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
    timesteps: int = 1000,
    t_samples: List[int] = None,  # List of timesteps to compute lambda max for
    max_iterations: int = 100,
    reltol: float = 1e-2,
    use_power_iteration: bool = False,
    use_ema: bool = False,
    verbose: bool = True,
    use_preconditioner: bool = False,
    optimizer_type: str = 'auto',
    learning_rate: Optional[float] = None,
    data_root: str = './data',
    data_seed: Optional[int] = 42
) -> dict:
    """
    Compute lambda max for all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Path to directory containing checkpoints
        device: Device to run computation on (default: cuda if available, else cpu)
        batch_size: Batch size for computing loss
        image_size: Image dimensions (channels, height, width)
        timesteps: Number of diffusion timesteps
        t_samples: List of timestep values to compute lambda max for (default: [49])
        max_iterations: Maximum iterations for eigenvalue computation
        reltol: Relative tolerance for convergence
        use_power_iteration: Use power iteration instead of LOBPCG
        use_ema: Load EMA weights instead of regular weights
        verbose: Print progress
        use_preconditioner: If True, compute preconditioned lambda max (P^{-1} H)
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'sgd', or 'auto' to infer)
        learning_rate: Learning rate for constant preconditioner (required if optimizer state not available)
        data_root: Root directory for CIFAR-10 dataset
        data_seed: Random seed for data sampling (for reproducibility)
        
    Returns:
        Dictionary mapping t_sample -> (epochs, lambda_max_values) tuples
    """
    if t_samples is None:
        t_samples = [49]  # Default to max timestep
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
        time_emb_dim=arch_params['time_emb_dim']
    )
    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    
    # Load real CIFAR-10 images for computing loss
    if verbose:
        print(f"Loading {batch_size} CIFAR-10 images from {data_root}...")
    x_sample = load_cifar10_sample(
        batch_size=batch_size,
        image_size=image_size[1:],  # (height, width) from (channels, height, width)
        data_root=data_root,
        seed=data_seed
    )
    x_sample = x_sample.to(device)
    if verbose:
        print(f"Loaded CIFAR-10 images with shape {x_sample.shape} and range [{x_sample.min():.3f}, {x_sample.max():.3f}]")
    
    # Validate t_samples are within valid range
    for t in t_samples:
        if t < 0 or t >= timesteps:
            raise ValueError(f"t_sample={t} is out of bounds. Valid range: [0, {timesteps-1}]")
    
    # Store results for each t_sample
    results = {t: {'epochs': [], 'lambda_max_values': []} for t in t_samples}
    
    # Process each checkpoint
    for checkpoint_path, epoch in tqdm(checkpoint_files, desc="Computing lambda max"):
        try:
            if verbose:
                print(f"\nProcessing checkpoint: {checkpoint_path.name} (epoch {epoch})")
            
            # Compute lambda max for each t_sample
            for t_sample in t_samples:
                try:
                    # Note: Preconditioner will be created inside compute_lambda_max_from_checkpoint_simple
                    # after the model is loaded, to ensure size matching
                    
                    # Compute lambda max for this checkpoint and timestep
                    lambda_max = compute_lambda_max_from_checkpoint_simple(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        schedule=schedule,
                        x_sample=x_sample,
                        t_sample=t_sample,
                        device=device,
                        use_ema=use_ema,
                        k=1,  # Only compute largest eigenvalue
                        max_iterations=max_iterations,
                        reltol=reltol,
                        use_power_iteration=use_power_iteration,
                        return_eigenvectors=False,
                        P=None,  # Will be created inside if use_preconditioner is True
                        use_preconditioner=use_preconditioner,
                        optimizer_type=optimizer_type,
                        learning_rate=learning_rate
                    )
                    
                    # Convert to Python float if it's a tensor
                    if isinstance(lambda_max, torch.Tensor):
                        lambda_max = lambda_max.item()
                    
                    results[t_sample]['epochs'].append(epoch)
                    results[t_sample]['lambda_max_values'].append(lambda_max)
                    
                    if verbose:
                        print(f"  t={t_sample}: Lambda max: {lambda_max:.6f}")
                        
                except Exception as e:
                    print(f"  Error processing t_sample={t_sample} for checkpoint {checkpoint_path.name}: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"  Error processing checkpoint {checkpoint_path.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Convert to format: {t_sample: (epochs, lambda_max_values)}
    return {t: (results[t]['epochs'], results[t]['lambda_max_values']) for t in t_samples}


def plot_lambda_max_history(
    results: dict,
    save_path: Optional[str] = None,
    title: str = "Lambda Max vs Epoch",
    is_preconditioned: bool = False
):
    """
    Plot lambda max values over epochs for multiple timesteps.
    
    Args:
        results: Dictionary mapping t_sample -> (epochs, lambda_max_values) tuples
        save_path: Path to save the plot (optional)
        title: Plot title
        is_preconditioned: If True, indicates this is preconditioned lambda max
    """
    plt.figure(figsize=(12, 7))
    
    # Plot each timestep as a separate line
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (t_sample, (epochs, lambda_max_values)) in enumerate(sorted(results.items())):
        plt.plot(epochs, lambda_max_values, 'o-', linewidth=2, markersize=5, 
                label=f't={t_sample}', color=colors[i])
    
    plt.xlabel('Epoch', fontsize=12)
    if is_preconditioned:
        plt.ylabel('Lambda Max (Largest Eigenvalue of P^{-1} H)', fontsize=12)
    else:
        plt.ylabel('Lambda Max (Largest Hessian Eigenvalue)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the script."""
    # Configuration

    checkpoint_dir = "tests/runs_ddim_test_progressive_difficulty_curriculum"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters for lambda max computation
    batch_size = 4
    image_size = (3, 32, 32)  # CIFAR-10
    timesteps = 1000
    t_samples = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]  # Multiple timesteps to compute lambda max for
    max_iterations = 100
    reltol = 1e-2
    use_power_iteration = False  # Use LOBPCG by default (faster)
    use_ema = False  # Set to True if you want to use EMA weights
    
    # Data settings
    data_root = './data'  # Root directory for CIFAR-10 dataset
    data_seed = 42  # Random seed for data sampling (for reproducibility)
    
    # Preconditioner settings
    use_preconditioner = True  # Set to True to compute preconditioned lambda max (P^{-1} H)
    optimizer_type = 'auto'  # 'auto', 'adam', 'rmsprop', or 'sgd'
    # Learning rate for constant preconditioner (required if optimizer state not available)
    # Set this to the learning rate you used during training:
    # - For SGD (edm_style_preconditioning.py): typically 1e-2
    # - For Adam (base_implementation.py): typically 2e-4
    # If your checkpoints have optimizer state, this will be ignored
    learning_rate = 2e-4
    
    print("=" * 60)
    print("Computing Lambda Max from Checkpoints")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Timesteps: {timesteps}")
    print(f"t_samples: {t_samples}")
    print(f"Max iterations: {max_iterations}")
    print(f"Relative tolerance: {reltol}")
    print(f"Use power iteration: {use_power_iteration}")
    print(f"Use EMA weights: {use_ema}")
    print(f"Data root: {data_root}")
    print(f"Data seed: {data_seed}")
    print(f"Use preconditioner: {use_preconditioner}")
    if use_preconditioner:
        print(f"Optimizer type: {optimizer_type}")
        print(f"Learning rate: {learning_rate if learning_rate else 'auto (from checkpoint)'}")
    print("=" * 60)
    
    # Compute lambda max for all checkpoints and timesteps
    results = compute_lambda_max_for_checkpoints(
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        timesteps=timesteps,
        t_samples=t_samples,
        max_iterations=max_iterations,
        reltol=reltol,
        use_power_iteration=use_power_iteration,
        use_ema=use_ema,
        verbose=True,
        use_preconditioner=use_preconditioner,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        data_root=data_root,
        data_seed=data_seed
    )
    
    if len(results) == 0:
        print("No successful computations. Exiting.")
        return
    
    # Print summary for each timestep
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for t_sample in sorted(results.keys()):
        epochs, lambda_max_values = results[t_sample]
        if len(epochs) > 0:
            print(f"\nt={t_sample}:")
            print(f"  Successfully computed lambda max for {len(epochs)} checkpoints")
            print(f"  Epoch range: {min(epochs)} - {max(epochs)}")
            print(f"  Lambda max range: {min(lambda_max_values):.6f} - {max(lambda_max_values):.6f}")
            print(f"  Mean lambda max: {np.mean(lambda_max_values):.6f}")
            print(f"  Std lambda max: {np.std(lambda_max_values):.6f}")
    
    # Save results to CSV (one file with all timesteps)
    suffix = "_preconditioned" if use_preconditioner else ""
    results_path = Path(checkpoint_dir) / f"lambda_max_results{suffix}.csv"
    import csv
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 't_sample', 'lambda_max'])
        for t_sample in sorted(results.keys()):
            epochs, lambda_max_values = results[t_sample]
            for epoch, lambda_max in zip(epochs, lambda_max_values):
                writer.writerow([epoch, t_sample, lambda_max])
    print(f"\nResults saved to: {results_path}")
    
    # Plot results (all timesteps on same plot)
    plot_path = Path(checkpoint_dir) / f"lambda_max_plot{suffix}.png"
    title_suffix = " (Preconditioned P^{-1} H)" if use_preconditioner else ""
    plot_lambda_max_history(
        results=results,
        save_path=str(plot_path),
        title=f"Lambda Max (Largest Hessian Eigenvalue){title_suffix} vs Epoch",
        is_preconditioned=use_preconditioner
    )


if __name__ == "__main__":
    main()

