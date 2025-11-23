"""
Implementation 6

Implementation: Stable Target Field (STF) for Reduced Variance
Reference: Xu et al., 2023

Summary of Changes from Standard Training:

1.  Stable Target Field (STF) Objective:
    To address the high variance of training targets in the "intermediate regime" of diffusion, this implementation replaces the standard Denoising Score Matching (DSM) objective with STF. This approach explicitly trades a small, asymptotically vanishing bias for significantly reduced variance in the gradient estimates, leading to improved stability and image quality.

2.  Weighted Targets via Batch-Reference Importance Sampling:
    Instead of regressing towards a single ground-truth noise instance, the model regresses towards a weighted average of the noise derived from the entire mini-batch.
    We implement this using self-normalized importance sampling, where importance weights are computed dynamically via pairwise Euclidean distances between noisy and clean samples, scaled by the noise variance $sigma_t^2$.

https://arxiv.org/abs/2302.00670
"""

from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    UNet, DiffusionSchedule, sample_loop)

def stf_loss(model, schedule, x_start, t):
    """
    Implements Stable Target Field (STF) loss.
    Calculates a weighted target based on the entire batch acting as the reference.
    """
    # 1. Sample noise and perturb image (Same as standard DSM)
    noise = torch.randn_like(x_start)
    x_noisy = schedule.q_sample(x_start=x_start, t=t, noise=noise)
    
    # 2. Prepare for Pairwise Distance Calculation
    # We need to calculate weights w_ij proportional to p(x_noisy_i | x_start_j)
    # This requires flattening the images to vectors
    B = x_start.shape[0]
    flat_noisy = x_noisy.view(B, -1)
    flat_start = x_start.view(B, -1)
    
    # 3. Get schedule parameters for the specific timesteps t
    # Reshape for broadcasting: [B, 1]
    alpha_bar = schedule.alphas_cumprod[t].view(-1, 1)
    sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[t].view(-1, 1)
    one_minus_alpha_bar = (1. - schedule.alphas_cumprod[t]).view(-1, 1) # This is sigma^2
    
    # 4. Calculate Squared Euclidean Distance Matrix
    # dist_sq[i, j] = ||x_noisy[i] - sqrt(alpha_bar[i]) * x_start[j]||^2
    # Expansion: ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
    
    # Force float32 for distance calculation to prevent overflow/underflow in AMP
    with torch.cuda.amp.autocast(enabled=False):
        flat_noisy_f32 = flat_noisy.float()
        flat_start_f32 = flat_start.float()
        
        norm_noisy_sq = (flat_noisy_f32**2).sum(dim=1, keepdim=True)  # [B, 1]
        norm_start_sq = (flat_start_f32**2).sum(dim=1).unsqueeze(0)   # [1, B]
        dot_prod = torch.mm(flat_noisy_f32, flat_start_f32.t())       # [B, B]
        
        # Note: We project x_start[j] by sqrt_alpha_bar[i] because we are checking 
        # likelihood of x_noisy[i] coming from x_start[j] at step t[i]
        dist_sq = norm_noisy_sq + (alpha_bar * norm_start_sq) - (2 * sqrt_alpha_bar * dot_prod)
        
        # 5. Calculate Importance Weights (Softmax)
        # logits = -dist_sq / (2 * sigma^2)
        # Add epsilon to sigma^2 to prevent division by zero at t=0
        logits = -dist_sq / (2 * one_minus_alpha_bar + 1e-5)
        weights = F.softmax(logits, dim=1) # [B, B] sum over columns (j) = 1

    # 6. Construct the Stable Target
    # The target noise for x_i coming from x_j is: (x_noisy_i - sqrt(alpha_i)*x_start_j) / sigma_i
    # The STF target is the weighted average of these noises.
    # STF_Target_i = (x_noisy_i - sqrt(alpha_i) * Weighted_Average_X_Start) / sigma_i
    
    # Compute weighted average of clean images: sum_j (w_ij * x_start_j)
    weights = weights.to(x_start.dtype) # Cast back to FP16 if needed
    weighted_x_start = torch.mm(weights, flat_start).view_as(x_start)
    
    # Calculate the STF noise target
    # Reshape scalars for image broadcasting [B, 1, 1, 1]
    sqrt_alpha_view = sqrt_alpha_bar.view(-1, 1, 1, 1)
    sigma_view = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    target_noise = (x_noisy - sqrt_alpha_view * weighted_x_start) / sigma_view

    # 7. Prediction and Loss
    predicted_noise = model(x_noisy, t)
    
    # Use MSE between predicted noise and the STF target
    return F.mse_loss(predicted_noise, target_noise)


def stf_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', ema_decay=0.995):
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
        ema_decay: EMA decay factor (None to disable EMA)
    """
    model = model.to(device)

    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), device=device, use_buffers=True)

    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
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
            t = torch.randint(0, schedule.timesteps, (b,), device=device).long()

            opt.zero_grad()

            # --- 2. Autocast Context ---
            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = stf_loss(model, schedule, x, t)

            # --- 3. Scale and Step ---
            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            ema_model.update_parameters(model)

            running_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.5f}'})

        avg_loss = running_loss / len(train_loader)
        print(f'End epoch {epoch+1}, avg loss {avg_loss:.4f}')

        # save checkpoint (model + ema)
        ckpt = {
            'model_state': model.state_dict(),
            'ema_state': ema_model.state_dict(), # Built-in state dict
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch+1
        }
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # sample and save a grid using EMA weights for nicer samples
        ema_model.eval()
        samples = sample_loop(ema_model, schedule, (16,3,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
