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


def adaptive_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs'):
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

    sampler_model = TimestepSampler(in_channels=3).to(device)
    sampler_lr = 1e-3
    sampler_opt = torch.optim.Adam(sampler_model.parameters(), lr=sampler_lr)
    sampler_freq = 40   # f_S: Update sampler every 40 steps
    subset_size = 3     # |S|: Size of timestep subset for approximation
    entropy_coef = 1e-2 # Entropy regularization coefficient

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_lr = 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader), eta_min=final_lr)
    scaler = torch.amp.GradScaler('cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        model.train()
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

            # Autocast Context
            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = p_losses(model, schedule, x, t)

            # Scale and Step
            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

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
            'sampler_state': sampler_model.state_dict(), # Save sampler
            'sampler_optimizer_state': sampler_opt.state_dict(), # Save optimizer
            'epoch': epoch+1,
            'avg_epoch_loss': avg_loss
        }
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        model.eval()
        samples = sample_loop(model, schedule, (16,3,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
