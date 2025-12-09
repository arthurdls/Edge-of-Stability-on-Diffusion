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
import csv

from implementations.base_implementation import (
    DiffusionSchedule, UNet, sample_loop, p_losses)

from torch.nn.attention import SDPBackend, sdpa_kernel
from implementations.utils.AEoSAnalyzer import AEoSAnalyzer

class CurriculumPacer:
    """
    Implements the Curriculum Learning pacing strategy described in 
    Algorithms 1 and 2 of the paper.

    Logic:
    1. Divides timesteps into N clusters.
    2. Starts with the easiest cluster (highest noise/timesteps).
    3. Expands to include harder clusters (lower timesteps) when loss converges.
    """
    def __init__(self, total_timesteps=1000, num_clusters=20, max_patience=200, smoothing=0.99, min_delta=1e-4, save_dir=None):
        self.total_timesteps = total_timesteps
        self.num_clusters = num_clusters
        self.max_patience = max_patience
        # Smoothing factor to stabilize batch loss for the pacing function
        self.smoothing = smoothing
        self.min_delta = min_delta

        self.save_dir = save_dir
        if self.save_dir:
            self.csv_path = Path(self.save_dir) / 'curriculum_log.csv'
            # Create file and write headers
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['global_step', 'stage'])

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

    def update(self, batch_loss, global_step):
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
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Transition Logic
        if self.patience_counter > self.max_patience:
            if self.current_stage < self.num_clusters:
                self.current_stage += 1
                print(f"\n[Curriculum] Converged. Advancing to Stage {self.current_stage}/{self.num_clusters}")
                if self.save_dir and global_step is not None:
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, self.current_stage])

                # Reset patience and best loss for the new stage (Algorithm 2)
                self.patience_counter = 0
                self.best_loss = float('inf') 
                # Optional: Reset running loss to adapt quickly to new task difficulty
                self.running_loss = None 


@torch.compiler.disable
def measure_aeos_step(model, opt, analyzer, global_step, lr, monitor_timesteps, schedule, device, x, t, current_avg_loss, measure_bs):
    """
    Helper function to run the AEoS measurement in strict eager mode.
    """

    # We still need the math backend for higher-order derivatives
    with sdpa_kernel([SDPBackend.MATH]):
        for ts_req in monitor_timesteps:
            torch.cuda.empty_cache()

            # 1. Prepare Data
            if ts_req == 'random':
                t_measure = t[:measure_bs]
                x_measure = x[:measure_bs]
                ts_label = "random"
            else:
                t_measure = torch.full((measure_bs,), ts_req, device=device).long()
                x_measure = x[:measure_bs]
                ts_label = f"t={ts_req}"

            # 2. Forward Pass (Eager Mode)
            loss_measure = p_losses(model, schedule, x_measure, t_measure)

            # 3. Analyze
            lmax = analyzer.log_step(
                model, opt, loss_measure, global_step, lr,
                timestep_label=ts_label, average_loss=current_avg_loss
            )

            print(f"  [{ts_label}] Lambda Max (P^-1 H): {lmax:.4f}")
            del loss_measure

    print(f"  Stability Threshold (38/Eta): {38.0/lr:.4f}")
    torch.cuda.empty_cache()


def progressive_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', measure_bs=16, checkpoint=None):
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
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if hasattr(torch, 'compile'):
        print("Compiling model...")
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    scaler = torch.amp.GradScaler('cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AEoSAnalyzer(save_dir)
    monitor_timesteps = ['random', 1, 100, 200, 300, 400, 500, 600, 700, 800, 999]

    # Curriculum Initialization
    # "We divided whole timesteps [0, T] into 10 uniformly divided intervals" 
    # "Maximum patience tau ... is a fixed hyper-parameter"
    pacer = CurriculumPacer(
        total_timesteps=schedule.timesteps,
        num_clusters=20,   # N=10
        max_patience=2,   # tau=2
        save_dir=save_dir
    )

    # Initialize starting epoch
    start_epoch = 0

    # --- Checkpoint Loading ---
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt_data = torch.load(checkpoint, map_location=device, weights_only=True)
        
        # Load model weights
        model.load_state_dict(ckpt_data['model_state'])
        
        # Load optimizer and scaler state
        opt.load_state_dict(ckpt_data['optimizer_state'])
        scaler.load_state_dict(ckpt_data['scaler_state'])
        
        # Restore epoch
        start_epoch = ckpt_data['epoch']
        
        # --- Restore Curriculum State ---
        pacer.current_stage = ckpt_data['curriculum_stage']
        print(f"Resumed Curriculum at Stage {pacer.current_stage}/{pacer.num_clusters}")
            
        # Recalculate the active timestep range for clarity
        current_min_t = pacer.get_min_timestep()
        print(f"Current training range: t ∈ [{current_min_t}, {schedule.timesteps}]")
        
        print(f"Resumed from epoch {start_epoch} (Prev Loss: {ckpt_data.get('avg_epoch_loss', 'N/A'):.4f})")

    global_step = 0
    interval_loss = 0.0
    for epoch in range(start_epoch, epochs):
        compiled_model.train()
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

            # --- AEoS Measurement Check ---
            if global_step > 0 and global_step % 1 == 0:
                print(f"\n[Step {global_step}] Measuring AEoS...")

                current_avg_loss = interval_loss / 1
                interval_loss = 0.0 # reset every 1 interval

                measure_aeos_step(
                    model, opt, analyzer, global_step, lr,
                    monitor_timesteps, schedule, device, x, t,
                    current_avg_loss, measure_bs
                )

            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = p_losses(compiled_model, schedule, x, t)

            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Curriculum Update
            # Update the pacer with current loss to check for convergence
            pacer.update(loss.item(), global_step)

            running_loss += loss.item()
            interval_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr}'})

        avg_loss = running_loss / len(train_loader)
        print(f'End epoch {epoch+1}, avg loss {avg_loss:.4f}')

        # save checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch+1,
            'curriculum_stage': pacer.current_stage, # Save curriculum state
            'avg_epoch_loss': avg_loss
        }
        # torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(ckpt, save_dir / f'latest.pt')

        compiled_model.eval()
        samples = sample_loop(compiled_model, schedule, (16,model.in_ch,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
