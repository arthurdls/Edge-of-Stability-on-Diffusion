"""
Script to load lambda max results from CSV and plot them, excluding t=0.

Usage:
    python plot_lambda_max_from_csv.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_lambda_max_from_csv(
    csv_path: str,
    save_path: str = None,
    title: str = "Lambda Max (Largest Eigenvalue of P^{-1} H) vs Epoch",
    exclude_zero: bool = True
):
    """
    Load lambda max results from CSV and plot them.
    Supports both base implementation (t_sample) and EDM (sigma_sample).
    
    Args:
        csv_path: Path to the CSV file
        save_path: Path to save the plot (optional)
        title: Plot title
        exclude_zero: If True, exclude t=0 or sigma=0 entries from the plot
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Detect which column exists (t_sample for base, sigma_sample for EDM)
    if 'sigma_sample' in df.columns:
        sample_col = 'sigma_sample'
        is_edm = True
        print(f"Detected EDM format (sigma_sample column)")
    elif 't_sample' in df.columns:
        sample_col = 't_sample'
        is_edm = False
        print(f"Detected base implementation format (t_sample column)")
    else:
        raise ValueError(f"CSV must have either 't_sample' or 'sigma_sample' column. Found columns: {df.columns.tolist()}")
    
    print(f"Unique samples: {sorted(df[sample_col].unique())}")
    
    # Filter out zero if requested
    if exclude_zero:
        df_filtered = df[df[sample_col] != 0].copy()
        print(f"After filtering zero: {len(df_filtered)} rows")
        print(f"Remaining samples: {sorted(df_filtered[sample_col].unique())}")
    else:
        df_filtered = df
    
    if len(df_filtered) == 0:
        print("No data to plot after filtering!")
        return
    
    # Group by sample column
    results = {}
    for sample_val in sorted(df_filtered[sample_col].unique()):
        sample_data = df_filtered[df_filtered[sample_col] == sample_val]
        epochs = sample_data['epoch'].values
        lambda_max_values = sample_data['lambda_max'].values
        results[sample_val] = (epochs, lambda_max_values)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Plot each sample value as a separate line
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (sample_val, (epochs, lambda_max_values)) in enumerate(sorted(results.items())):
        if is_edm:
            label = f'σ={sample_val:.3f}' if isinstance(sample_val, float) else f'σ={sample_val}'
        else:
            label = f't={sample_val}'
        plt.plot(epochs, lambda_max_values, 'o-', linewidth=2, markersize=5, 
                label=label, color=colors[i])
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Lambda Max (Largest Eigenvalue of P^{-1} H)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_loss_from_csv(
    csv_path: str,
    save_path: str = None,
    title: str = "Loss vs Epoch",
    plot_step_loss: bool = True,
    plot_avg_loss: bool = True,
    plot_lr: bool = False
):
    """
    Load loss results from CSV and plot them.
    
    Args:
        csv_path: Path to the CSV file
        save_path: Path to save the plot (optional)
        title: Plot title
        plot_step_loss: If True, plot step_loss
        plot_avg_loss: If True, plot avg_loss
        plot_lr: If True, plot learning rate on secondary y-axis
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    if len(df) == 0:
        print("No data to plot!")
        return
    
    # Extract data
    epochs = df['epoch'].values
    
    # Handle both 'lr' and 'learning_rate' column names
    lr_col = None
    if 'learning_rate' in df.columns:
        lr_col = 'learning_rate'
    elif 'lr' in df.columns:
        lr_col = 'lr'
    
    # Create figure
    if plot_lr and lr_col is not None:
        fig, ax1 = plt.subplots(figsize=(12, 7))
    else:
        plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
    
    # Plot step_loss if requested
    if plot_step_loss and 'step_loss' in df.columns:
        ax1.plot(epochs, df['step_loss'].values, 'o-', linewidth=2, markersize=4, 
                label='Step Loss', color='blue', alpha=0.7)
    
    # Plot avg_loss if requested
    if plot_avg_loss and 'avg_loss' in df.columns:
        ax1.plot(epochs, df['avg_loss'].values, 's-', linewidth=2, markersize=4, 
                label='Average Loss', color='red', alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate on secondary y-axis if requested
    if plot_lr and lr_col is not None:
        ax2 = ax1.twinx()
        color_lr = 'green'
        ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=12)
        ax2.plot(epochs, df[lr_col].values, '^-', linewidth=2, markersize=4, 
                label='Learning Rate', color=color_lr, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color_lr)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    else:
        ax1.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_combined_metrics(
    loss_csv_path: str,
    lambda_max_csv_path: str,
    save_path: str = None,
    title: str = "Learning Rate and Average Lambda Max (Sharpness) vs Epoch",
    exclude_zero: bool = True
):
    """
    Plot learning rate and average lambda max (sharpness) across all samples on the same plot.
    Adds vertical dashed lines at learning rate drops.
    Supports both base implementation (t_sample) and EDM (sigma_sample).
    
    Args:
        loss_csv_path: Path to the loss CSV file (with epoch, learning_rate)
        lambda_max_csv_path: Path to the lambda max CSV file (with epoch, t_sample/sigma_sample, lambda_max)
        save_path: Path to save the plot (optional)
        title: Plot title
        exclude_zero: If True, exclude t=0 or sigma=0 entries from lambda max
    """
    # Load loss data (for learning rate)
    loss_df = pd.read_csv(loss_csv_path)
    loss_df = loss_df.sort_values('epoch')
    print(f"Loaded {len(loss_df)} rows from {loss_csv_path}")
    
    # Load lambda max data
    lambda_df = pd.read_csv(lambda_max_csv_path)
    print(f"Loaded {len(lambda_df)} rows from {lambda_max_csv_path}")
    
    # Detect which column exists
    if 'sigma_sample' in lambda_df.columns:
        sample_col = 'sigma_sample'
        is_edm = True
        print(f"Detected EDM format (sigma_sample column)")
    elif 't_sample' in lambda_df.columns:
        sample_col = 't_sample'
        is_edm = False
        print(f"Detected base implementation format (t_sample column)")
    else:
        raise ValueError(f"Lambda max CSV must have either 't_sample' or 'sigma_sample' column. Found columns: {lambda_df.columns.tolist()}")
    
    # Filter lambda max data
    if exclude_zero:
        lambda_df_filtered = lambda_df[lambda_df[sample_col] != 0].copy()
    else:
        lambda_df_filtered = lambda_df.copy()
    
    if len(lambda_df_filtered) == 0:
        print("No lambda max data to plot after filtering!")
        return
    
    # Compute average lambda max across all samples for each epoch
    lambda_avg = lambda_df_filtered.groupby('epoch')['lambda_max'].mean().reset_index()
    lambda_avg.columns = ['epoch', 'avg_lambda_max']
    lambda_avg = lambda_avg.sort_values('epoch')
    
    available_samples = sorted(lambda_df_filtered[sample_col].unique())
    sample_type = "sigma values" if is_edm else "timesteps"
    print(f"Available {sample_type}: {available_samples}")
    print(f"Computing average sharpness across {len(available_samples)} {sample_type}")
    
    # Merge data on epoch
    merged_df = loss_df.merge(lambda_avg, on='epoch', how='inner')
    
    if len(merged_df) == 0:
        print("No overlapping epochs between loss and lambda max data!")
        return
    
    print(f"Plotting {len(merged_df)} data points")
    
    # Handle both 'lr' and 'learning_rate' column names
    if 'learning_rate' in merged_df.columns:
        lr_col = 'learning_rate'
    elif 'lr' in merged_df.columns:
        lr_col = 'lr'
    else:
        raise ValueError(f"Loss CSV must have either 'lr' or 'learning_rate' column. Found columns: {loss_df.columns.tolist()}")
    
    # Detect learning rate drops
    lr_drops = []
    for i in range(1, len(merged_df)):
        if merged_df.iloc[i][lr_col] < merged_df.iloc[i-1][lr_col]:
            lr_drops.append(merged_df.iloc[i]['epoch'])
    
    print(f"Found {len(lr_drops)} learning rate drops at epochs: {lr_drops}")
    
    # Create figure with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    epochs = merged_df['epoch'].values
    
    # Plot learning rate on left y-axis
    color_lr = 'green'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Learning Rate', color=color_lr, fontsize=12)
    line1 = ax1.plot(epochs, merged_df[lr_col].values, 'o-', 
                     color=color_lr, linewidth=2, markersize=4, alpha=0.7, label='Learning Rate')
    ax1.tick_params(axis='y', labelcolor=color_lr)
    ax1.grid(True, alpha=0.3)
    
    # Plot average lambda max on right y-axis
    ax2 = ax1.twinx()
    color_lambda = 'blue'
    ax2.set_ylabel('Average Lambda Max (Sharpness)', color=color_lambda, fontsize=12)
    # Use the sample_type determined earlier
    line2 = ax2.plot(epochs, merged_df['avg_lambda_max'].values, '^-', 
                     color=color_lambda, linewidth=2, markersize=4, alpha=0.7, 
                     label=f'Average Lambda Max (across all {sample_type})')
    ax2.tick_params(axis='y', labelcolor=color_lambda)
    
    # Add vertical dashed lines at learning rate drops
    if len(lr_drops) > 0:
        for drop_epoch in lr_drops:
            ax1.axvline(x=drop_epoch, color='grey', linestyle='--', linewidth=1.0, 
                       alpha=0.6, label='LR Drop' if drop_epoch == lr_drops[0] else '')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # Add LR drop to legend if present
    if len(lr_drops) > 0:
        from matplotlib.lines import Line2D
        lr_drop_line = Line2D([0], [0], color='grey', linestyle='--', linewidth=1.5, alpha=0.6)
        lines.append(lr_drop_line)
        labels.append('LR Drop')
    ax1.legend(lines, labels, loc='best', fontsize=10)
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the script."""
    # Configuration: Choose which implementation to plot
    # Options: 'progressive_curriculum', 'base', 'edm', 'min_snr'
    implementation = 'progressive_curriculum'  # Change this to switch implementations
    
    # Set paths based on implementation
    if implementation == 'progressive_curriculum':
        base_dir = "tests/runs_ddim_test_progressive_difficulty_curriculum"
        loss_csv = f"{base_dir}/progressive_metrics.csv"  # Progressive curriculum metrics CSV
        plot_title_suffix = " (Progressive Curriculum)"
    elif implementation == 'base':
        base_dir = "tests/runs_ddim_test_base_implementation"
        loss_csv = None  # Base implementation may not save loss CSV
        plot_title_suffix = " (Base Implementation)"
    elif implementation == 'edm':
        base_dir = "tests/runs_ddim_test_edm_preconditioning"
        loss_csv = f"{base_dir}/edm_loss_lr.csv"
        plot_title_suffix = " (EDM Preconditioning)"
    elif implementation == 'min_snr':
        base_dir = "tests/runs_ddim_test_min_snr_reweighting"
        loss_csv = None
        plot_title_suffix = " (Min SNR Reweighting)"
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    # Try to find lambda max CSV (check both preconditioned and non-preconditioned)
    output_dir = Path(base_dir)
    lambda_max_csv_precond = output_dir / "lambda_max_results_preconditioned.csv"
    lambda_max_csv_regular = output_dir / "lambda_max_results.csv"
    
    if lambda_max_csv_precond.exists():
        lambda_max_csv = str(lambda_max_csv_precond)
        is_preconditioned = True
        save_path = str(output_dir / "lambda_max_plot_preconditioned.png")
    elif lambda_max_csv_regular.exists():
        lambda_max_csv = str(lambda_max_csv_regular)
        is_preconditioned = False
        save_path = str(output_dir / "lambda_max_plot.png")
    else:
        print(f"Warning: Lambda max CSV not found in {base_dir}")
        print("  Checked for:")
        print(f"    - {lambda_max_csv_precond}")
        print(f"    - {lambda_max_csv_regular}")
        print("Please run compute_sharpness_progressive.py first to generate the CSV file.")
        return
    
    print("=" * 60)
    print("Plotting Lambda Max from CSV")
    print("=" * 60)
    print(f"Implementation: {implementation}")
    print(f"CSV file: {lambda_max_csv}")
    print(f"Preconditioned: {is_preconditioned}")
    print(f"Output plot: {save_path}")
    print("=" * 60)
    
    # Plot the results
    if is_preconditioned:
        title = f"Largest Eigenvalue of P^{{-1}} H vs Epoch{plot_title_suffix}"
    else:
        title = f"Largest Hessian Eigenvalue vs Epoch{plot_title_suffix}"
    
    plot_lambda_max_from_csv(
        csv_path=lambda_max_csv,
        save_path=save_path,
        title=title,
        exclude_zero=False
    )
    
    # Plot loss data if available
    if loss_csv is not None and Path(loss_csv).exists():
        loss_save_path = str(output_dir / "loss_plot.png")
        
        print("\n" + "=" * 60)
        print("Plotting Loss from CSV")
        print("=" * 60)
        print(f"CSV file: {loss_csv}")
        print(f"Output plot: {loss_save_path}")
        print("=" * 60)
        
        plot_loss_from_csv(
            csv_path=loss_csv,
            save_path=loss_save_path,
            title=f"Average Loss and Learning Rate vs Epoch{plot_title_suffix}",
            plot_step_loss=False,
            plot_avg_loss=True,
            plot_lr=True  # Also plot learning rate
        )
        
        # Plot combined metrics
        combined_save_path = str(output_dir / "combined_metrics_plot.png")
        
        print("\n" + "=" * 60)
        print("Plotting Combined Metrics (LR and Lambda Max)")
        print("=" * 60)
        print(f"Loss CSV: {loss_csv}")
        print(f"Lambda Max CSV: {lambda_max_csv}")
        print(f"Output plot: {combined_save_path}")
        print("=" * 60)
        
        combined_title = f"Learning Rate and Average Lambda Max (Sharpness) vs Epoch{plot_title_suffix}"
        if is_preconditioned:
            combined_title = f"Learning Rate and Average Lambda Max (P^{{-1}} H) vs Epoch{plot_title_suffix}"
        
        plot_combined_metrics(
            loss_csv_path=loss_csv,
            lambda_max_csv_path=lambda_max_csv,
            save_path=combined_save_path,
            title=combined_title,
            exclude_zero=False
        )
    else:
        if loss_csv is None:
            print(f"\nNote: Loss CSV not configured for {implementation} implementation.")
        else:
            print(f"\nNote: Loss CSV not found at {loss_csv}. Skipping loss and combined plots.")


if __name__ == "__main__":
    main()

