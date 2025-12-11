"""
Implementation 5

Adaptive Non-uniform Timestep Sampling for Standard Diffusion (DDPM)
Reference: Kim et al., 2025

Summary of Changes from Standard Training:

1.  Parameterized Beta Sampling:
    Instead of the fixed uniform sampling used in standard DDPM, this implementation utilizes a learnable timestep sampler applied to DDIM.
    The sampler is parameterized as a neural network that predicts $alpha$ and $beta$ values for a Beta distribution based on the input $x_0$, allowing the model to account for correlations between similar timesteps.

2.  Loss-Reduction Reward Mechanism:
    The sampling strategy is not heuristic or fixed; rather, the sampler is explicitly trained to prioritize timesteps that yield the greatest reduction in the variational lower bound (VLB).
    The objective is to maximize the "reward" $\Delta_{k}^{t}$, defined as the difference in diffusion loss before and after a gradient update at step $k$. Note that we still use MSE Loss.

3.  Efficient Look-Ahead Approximation:
    To make the reward calculation computationally feasible, the implementation approximates the true loss reduction using a "look-ahead" strategy on a small subset of timesteps $S$ (where $|S|=3$).
    Instead of evaluating the full timestep range, the algorithm computes the loss change only on this subset to estimate the optimal sampling direction.

4.  Online Policy Gradient Optimization:
    The sampler is updated jointly with the diffusion model in an online manner, utilizing a REINFORCE-style policy gradient update.
    To prevent premature convergence to a single timestep, the loss function includes an entropy regularization term.
    Updates to the sampler occur periodically (every $f_S$ steps) rather than at every iteration to minimize computational overhead.

https://arxiv.org/abs/2411.09998
"""
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    UNet, DiffusionSchedule, sample_loop, p_losses)

from torch.nn.attention import SDPBackend, sdpa_kernel
from implementations.utils.AEoSAnalyzer import AEoSAnalyzer


class TimestepSampler(nn.Module):
    """
    Adaptive Timestep Sampler as proposed in 'Adaptive Non-uniform Timestep Sampling'.
    Predicts alpha and beta parameters for a Beta distribution based on input x0.
    """
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        # Lightweight encoder as suggested in Appendix Table 7 
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus() # Ensures alpha and beta are positive 
        )

    def forward(self, x):
        # Returns parameters alpha, beta > 0
        params = self.net(x) + 1e-4 # Add epsilon for numerical stability
        return params[:, 0], params[:, 1]

    def get_dist(self, x):
        alpha, beta = self.forward(x)
        return torch.distributions.Beta(alpha, beta)


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


def adaptive_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', measure_bs=16, checkpoint=None):
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

    sampler_model = TimestepSampler(in_channels=3).to(device)
    sampler_lr = 1e-3
    sampler_opt = torch.optim.Adam(sampler_model.parameters(), lr=sampler_lr)
    sampler_freq = 40   # f_S: Update sampler every 40 steps
    subset_size = 3     # |S|: Size of timestep subset for approximation
    entropy_coef = 1e-2 # Entropy regularization coefficient

    scaler = torch.amp.GradScaler('cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AEoSAnalyzer(save_dir)
    monitor_timesteps = ['random', 1, 100, 200, 300, 400, 500, 600, 700, 800, 999]

    # Initialize starting epoch
    start_epoch = 0

    # --- Checkpoint Loading ---
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt_data = torch.load(checkpoint, map_location=device, weights_only=True)
        
        # Load main model states
        model.load_state_dict(ckpt_data['model_state'])
        opt.load_state_dict(ckpt_data['optimizer_state'])
        scaler.load_state_dict(ckpt_data['scaler_state'])
        
        # Load Sampler states (Crucial for resuming the learned policy)
        sampler_model.load_state_dict(ckpt_data['sampler_state'])
        sampler_opt.load_state_dict(ckpt_data['sampler_optimizer_state'])
        print("Resumed Timestep Sampler policy.")
        
        # Restore epoch
        start_epoch = ckpt_data['epoch']
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

            # Adaptive Sampling
            # Sample t from the learned Beta distribution
            dist = sampler_model.get_dist(x)
            t_float = dist.sample() # Values in [0, 1]

            # Store log_prob for policy gradient update later
            log_prob = dist.log_prob(t_float).mean()

            # Map [0, 1] to integer timesteps [0, T-1]
            t = (t_float * (schedule.timesteps - 1)).long().to(device)

            # Look-Ahead Preparation (Algorithm 2)
            # To approximate Delta (reward), we need loss *before* update on a subset S.
            # Only done every f_S steps to save compute.
            should_update_sampler = (global_step % sampler_freq == 0) and (global_step > 0)

            loss_S_before = 0
            subset_timesteps = None

            if should_update_sampler:
                # Select random subset S (Paper uses feature selection, but notes |S|=3 is robust)
                subset_timesteps = torch.randint(0, schedule.timesteps, (b, subset_size), device=device).long()

                # Compute Loss on S *before* update (using current model state)
                with torch.no_grad():
                    # We calculate loss for each timestep in S. 
                    # We define reward based on sum of losses in S.
                    for j in range(subset_size):
                        t_s = subset_timesteps[:, j]
                        # Using Mixed Precision for efficiency
                        with torch.amp.autocast('cuda'):
                            l_pre = p_losses(model, schedule, x, t_s)
                        loss_S_before += l_pre

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

            # Sampler Update
            if should_update_sampler:
                loss_S_after = 0
                
                # Compute Loss on S *after* diffusion model update
                with torch.no_grad():
                    for j in range(subset_size):
                        t_s = subset_timesteps[:, j]
                        with torch.amp.autocast('cuda'):
                            l_post = p_losses(model, schedule, x, t_s)
                        loss_S_after += l_post

                # Reward = Reduction in loss (Loss_before - Loss_after) 
                # We average over batch and subset
                reward = (loss_S_before - loss_S_after) / subset_size

                # Policy Gradient Loss: Maximize reward => Minimize -1 * reward * log_prob
                # Plus Entropy Regularization to prevent premature convergence
                entropy = dist.entropy().mean()
                sampler_loss = -(reward * log_prob) - (entropy_coef * entropy)

                sampler_opt.zero_grad()
                sampler_loss.backward()
                sampler_opt.step()
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
            'sampler_state': sampler_model.state_dict(), # Save sampler
            'sampler_optimizer_state': sampler_opt.state_dict(), # Save optimizer
            'epoch': epoch+1,
            'avg_epoch_loss': avg_loss
        }
        # torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(ckpt, save_dir / f'latest.pt')

        compiled_model.eval()
        samples = sample_loop(compiled_model, schedule, (16,model.in_ch,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
