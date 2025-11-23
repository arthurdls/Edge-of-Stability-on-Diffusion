"""
Implementation 4

Progressive Task Difficulty Curriculum Implementation for Standard Diffusion (DiT/DDPM)
Reference: Kim et al., 2025

Summary of Changes from Standard Training:

1.  Uniform Interval Clustering:
    Instead of the SNR-based/Quantile partitioning used for EDM, this implementation divides the discrete timestep range $[0, T]$ into $N$ uniformly sized intervals.
    Clusters are defined as $C_{i}=[frac{(i-1) cdot T}{N},frac{i cdot T}{N}]$.
    High indices correspond to large timesteps (high noise/easier tasks), while low indices correspond to small timesteps (low noise/harder tasks).

2.  Curriculum Sampling Strategy (Easy-to-Hard Accumulation):
    Training does not sample uniformly from $[0, T]$ initially. Instead, it samples from a restricted range based on the current curriculum stage.
    Training starts with the easiest cluster (highest timesteps) and progressively accumulates harder clusters (lower timesteps).
    At curriculum stage $n$, the model trains on the union of clusters $bigcup_{j=N-(n-1)}^{N}C_{j}$.

3.  Adaptive Pacing Function:
    This replaces fixed training durations with a dynamic schedule based on model performance.
    The transition between curriculum stages is determined by loss convergence.
    A "patience" hyperparameter ($tau$) tracks consecutive iterations without loss improvement; if the patience limit is reached, the curriculum expands to include the next harder cluster of timesteps.

4.  Convergence Phase:
    After the curriculum progresses through all $N$ stages (accumulating all clusters from easy to hard), the model continues training on the fully unrestricted timestep range (Standard Diffusion training) to reach final convergence.

https://arxiv.org/abs/2403.10348
"""
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    DiffusionSchedule, UNet, sample_loop, p_losses)


class CurriculumPacer:
    """
    Implements the Curriculum Learning pacing strategy described in 
    Algorithms 1 and 2 of the paper.

    Logic:
    1. Divides timesteps into N clusters.
    2. Starts with the easiest cluster (highest noise/timesteps).
    3. Expands to include harder clusters (lower timesteps) when loss converges.
    """
    def __init__(self, total_timesteps=1000, num_clusters=20, max_patience=200, smoothing=0.99):
        self.total_timesteps = total_timesteps
        self.num_clusters = num_clusters
        self.max_patience = max_patience
        # Smoothing factor to stabilize batch loss for the pacing function
        self.smoothing = smoothing

        # Curriculum state
        self.current_stage = 1 # Starts at stage 1 (easiest)
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.running_loss = None

        # Calculate cluster size (assuming uniform clustering for minimum change)
        self.cluster_size = total_timesteps // num_clusters

    def get_min_timestep(self):
        """
        Returns the minimum timestep index for the current stage.
        As stage increases, min_timestep decreases (adding harder, lower-noise tasks).
        Ref: "we jointly train the model with denoising tasks sampled from the 
        clusters U_{j=N-(n-1)}^{N} C_j".
        """
        # If we are at the final stage or beyond, sample full range
        if self.current_stage >= self.num_clusters:
            return 0

        # Calculate how many clusters are active.
        # Stage 1: Top 1 cluster active. 
        # Min t = Total - (1 * Size)
        cutoff = self.total_timesteps - (self.current_stage * self.cluster_size)
        return max(0, int(cutoff))

    def update(self, batch_loss):
        """
        Updates the pacer based on Algorithm 1: Pacing Function.
        """
        # Smooth the loss to prevent erratic jumps due to batch variance
        if self.running_loss is None:
            self.running_loss = batch_loss
        else:
            self.running_loss = self.smoothing * self.running_loss + (1 - self.smoothing) * batch_loss

        current_loss = self.running_loss

        # Check convergence criteria
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Transition Logic
        if self.patience_counter > self.max_patience:
            if self.current_stage < self.num_clusters:
                self.current_stage += 1
                print(f"\n[Curriculum] Converged. Advancing to Stage {self.current_stage}/{self.num_clusters}")

                # Reset patience and best loss for the new stage (Algorithm 2)
                self.patience_counter = 0
                self.best_loss = float('inf') 
                # Optional: Reset running loss to adapt quickly to new task difficulty
                self.running_loss = None 



def progressive_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs'):
    """
    Training loop for diffusion model.

    Args:
        model: The diffusion model (e.g., TinyUNet)
        schedule: DiffusionSchedule instance
        train_loader: DataLoader for training data
        device: torch.device
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints and samples
    """
    model = model.to(device)

    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_lr = 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader), eta_min=final_lr)
    scaler = torch.amp.GradScaler('cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum Initialization
    # "We divided whole timesteps [0, T] into 20 uniformly divided intervals" 
    # "Maximum patience tau ... is a fixed hyper-parameter"
    pacer = CurriculumPacer(
        total_timesteps=schedule.timesteps,
        num_clusters=20,   # N=20
        max_patience=200   # tau=200
    )

    global_step = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        running_loss = 0.0
        for i, (x, _) in enumerate(pbar):
            x = x.to(device)
            b = x.shape[0]

            # Curriculum Sampling
            # Determine the lower bound for t based on current stage
            min_t = pacer.get_min_timestep()
            # Sample t from [min_t, timesteps)
            # "Sampled from the clusters U ... C_j" 
            t = torch.randint(min_t, schedule.timesteps, (b,), device=device).long()

            opt.zero_grad()

            # --- 2. Autocast Context ---
            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = p_losses(model, schedule, x, t)

            # --- 3. Scale and Step ---
            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # Curriculum Update
            # Update the pacer with current loss to check for convergence
            pacer.update(loss.item())

            running_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.5f}'})

        avg_loss = running_loss / len(train_loader)
        print(f'End epoch {epoch+1}, avg loss {avg_loss:.4f}')

        # save checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch+1,
            'curriculum_stage': pacer.current_stage # Save curriculum state
        }
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        model.eval()
        samples = sample_loop(model, schedule, (16,3,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
