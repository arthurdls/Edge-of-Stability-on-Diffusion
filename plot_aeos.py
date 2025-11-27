"""
Visualization utilities for lambda max (Hessian eigenvalue) and Batch Sharpness tracking.
Inspired by standard plotting styles for Edge of Stability research.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

def plot_aeos_metrics(
    data: Union[str, pd.DataFrame],
    save_dir: Optional[str] = None,
    figsize=(12, 7),
    style="whitegrid"
):
    """
    Generate comprehensive plots for AEoS metrics: Batch Sharpness, Lambda Max, and Loss.
    
    Args:
        data: Path to CSV file or pandas DataFrame containing the logs.
        save_dir: Directory to save the plots. If None, plots are shown but not saved.
        figsize: Figure size tuple (width, height).
        style: Seaborn plot style.
    """
    # 1. Load Data
    if isinstance(data, (str, Path)):
        print(f"Loading data from {data}...")
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            print("Error: File not found.")
            return
    else:
        df = data

    # Ensure numeric types
    numeric_cols = ['step', 'batch_sharpness', 'lambda_max', 'lr', 
                    'stability_threshold_38_over_eta', 'average_loss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Setup Plot Style
    sns.set_theme(style=style)
    
    # Helper to sort timesteps logically
    def sort_key(x):
        if x == 'random': return -1
        try:
            return int(x.split('=')[1])
        except:
            return 9999

    try:
        sorted_timesteps = sorted(df['timestep'].unique(), key=sort_key)
    except:
        sorted_timesteps = df['timestep'].unique()

    # Extract threshold for plotting
    if 'stability_threshold_38_over_eta' in df.columns:
        threshold_df = df[['step', 'stability_threshold_38_over_eta']].drop_duplicates().sort_values('step')
    else:
        threshold_df = None

    # --- Plot 1: Batch Sharpness vs Iterations ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df, 
        x='step', 
        y='batch_sharpness', 
        hue='timestep', 
        hue_order=sorted_timesteps,
        marker='o',
        markersize=4,
        palette='viridis',
        ax=ax1
    )
    
    ax1.set_title('Batch Sharpness vs Iterations', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel(r'Batch Sharpness ($g^T H g / ||g||^2$)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Timestep")
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'batch_sharpness.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.show()

    # --- Plot 2: Lambda Max (Specific Timesteps) vs Iterations ---
    df_no_random = df[df['timestep'] != 'random'].copy()
    if not df_no_random.empty:
        # Sort remaining timesteps
        sorted_ts_no_rand = [ts for ts in sorted_timesteps if ts != 'random']
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        sns.lineplot(
            data=df_no_random, 
            x='step', 
            y='lambda_max', 
            hue='timestep', 
            hue_order=sorted_ts_no_rand,
            marker='o',
            markersize=4,
            palette='magma',
            ax=ax2
        )
        
        # Plot Threshold
        if threshold_df is not None:
            ax2.plot(
                threshold_df['step'], 
                threshold_df['stability_threshold_38_over_eta'], 
                color='red', 
                linestyle='--', 
                linewidth=2.5, 
                label=r'Stability Threshold ($38/\eta$)'
            )

        ax2.set_title(r'Lambda Max ($\lambda_{max}(P^{-1}H)$) - Specific Timesteps', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel(r'$\lambda_{max}$', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Timestep")
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'lambda_max_timesteps.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.show()

    # --- Plot 3: Lambda Max (Random Timestep) vs Iterations ---
    df_random = df[df['timestep'] == 'random'].copy()
    if not df_random.empty:
        fig3, ax3 = plt.subplots(figsize=figsize)
        sns.lineplot(
            data=df_random, 
            x='step', 
            y='lambda_max', 
            marker='o', 
            markersize=5,
            color='blue',
            label='Random Timestep $\lambda_{max}$',
            ax=ax3
        )
        
        # Plot Threshold
        if threshold_df is not None:
            ax3.plot(
                threshold_df['step'], 
                threshold_df['stability_threshold_38_over_eta'], 
                color='red', 
                linestyle='--', 
                linewidth=2.5, 
                label=r'Stability Threshold ($38/\eta$)'
            )

        ax3.set_title(r'Lambda Max ($\lambda_{max}(P^{-1}H)$) - Random Timestep', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel(r'$\lambda_{max}$', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'lambda_max_random.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.show()

    # --- Plot 4: Average Loss vs Iterations ---
    if 'average_loss' in df.columns:
        # Loss is duplicated per timestep row, so we drop duplicates
        df_loss = df[['step', 'average_loss']].drop_duplicates().sort_values('step')
        
        fig4, ax4 = plt.subplots(figsize=figsize)
        
        # Plot on Log Scale if loss varies greatly
        ax4.plot(df_loss['step'], df_loss['average_loss'], 'g-', linewidth=2, marker='o', markersize=4, label='MSE Loss')
        
        ax4.set_title('Average Training Loss', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Training Step', fontsize=12)
        ax4.set_ylabel('MSE Loss', fontsize=12, color='green')
        ax4.tick_params(axis='y', labelcolor='green')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')
        
        # Optional: Add log scale if range is large
        if df_loss['average_loss'].max() / (df_loss['average_loss'].min() + 1e-9) > 100:
            ax4.set_yscale('log')
            ax4.set_ylabel('MSE Loss (Log Scale)', fontsize=12, color='green')

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'loss_vs_iters.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.show()

if __name__ == "__main__":
    import argparse
    
    # Simple CLI
    parser = argparse.ArgumentParser(description="Plot AEoS metrics from CSV log.")
    parser.add_argument("csv_path", type=str, help="Path to the aeos_log.csv file")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save plots")
    
    # Try/Except block to allow running without args (defaults to local file)
    try:
        args = parser.parse_args()
        csv_path = args.csv_path
        save_dir = args.save_dir
    except:
        csv_path = "aeos_log.csv"
        save_dir = "."
        print(f"No arguments provided. Defaulting to '{csv_path}' in current directory.")

    if Path(csv_path).exists():
        plot_aeos_metrics(csv_path, save_dir)
    else:
        print(f"Error: '{csv_path}' not found.")