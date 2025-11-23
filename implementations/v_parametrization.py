"""
Implementation 2

Progressive Distillation for Fast Sampling of Diffusion Models
Reference: Salimans & Ho, 2022

Summary of Changes from Standard DDPM/DDIM:

1.  V-Parameterization (Prediction Target):
    - Instead of predicting noise ($epsilon$) or image ($x_0$), the model predicts velocity ($v$).
    - Formula: $v_t equiv alpha_t epsilon - sigma_t x$.
    - This ensures stability when the Signal-to-Noise Ratio (SNR) approaches zero (e.g., at $t=1$),
      where standard $epsilon$-prediction becomes unstable.

https://arxiv.org/abs/2202.00512
"""
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from implementations.base_implementation import (
    DiffusionSchedule, UNet)
torch.backends.cudnn.benchmark = True


# 1. Modified Loss Function: Predicts v instead of noise
def v_param_loss(model, schedule, x_start, t):
    noise = torch.randn_like(x_start)
    
    # Extract alpha (signal) and sigma (noise) scale factors
    # shape: [B, 1, 1, 1] for broadcasting
    sqrt_alpha = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_sigma = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    # z_t = alpha * x + sigma * eps (Standard forward process)
    x_noisy = sqrt_alpha * x_start + sqrt_sigma * noise
    
    # Calculate the v-target
    # Definition: v = alpha * eps - sigma * x
    v_target = sqrt_alpha * noise - sqrt_sigma * x_start
    
    # Model output is now interpreted as v
    predicted_v = model(x_noisy, t)
    
    # MSE loss against v_target
    return F.mse_loss(predicted_v, v_target)


# 2. Modified Sampling Function: Converts v back to x0/eps for the update step
@torch.no_grad()
def v_param_sample(model, schedule, x, t, t_prev, eta=0.0):
    """
    t: current timestep
    t_prev: next timestep
    """
    # 1. Get alphas/sigmas for the CURRENT step (t)
    alpha_bar_t = schedule.alphas_cumprod[t]
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)

    # 2. Get alphas/sigmas for the PREVIOUS step (t_prev)
    alpha_bar_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
    
    # 3. Model Prediction (Output is v)
    t_tensor = torch.tensor([t], device=x.device).repeat(x.shape[0])
    predicted_v = model(x, t_tensor)

    # 4. Convert v-prediction to x0 and epsilon
    # From paper: x0 = alpha * z - sigma * v
    x0_pred = sqrt_alpha_bar_t * x - sqrt_one_minus_alpha_bar_t * predicted_v
    
    # From algebra: eps = alpha * v + sigma * z
    predicted_noise = sqrt_alpha_bar_t * predicted_v + sqrt_one_minus_alpha_bar_t * x
    
    # Clamp x0 for stability (optional but recommended)
    x0_pred = x0_pred.clamp(-1., 1.)

    # --- Standard DDIM Update Step using the derived x0 and eps ---
    
    # Sigma calculation for stochastic DDIM (eta > 0)
    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
    
    # Direction pointing to x_t
    pred_dir_xt = torch.sqrt(torch.clamp(1. - alpha_bar_prev - sigma_t**2, min=0.0)) * predicted_noise
    
    # Compute x_{t-1}
    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + pred_dir_xt
    
    # Add noise if eta > 0
    if eta > 0:
        noise = torch.randn_like(x)
        x_prev = x_prev + sigma_t * noise
        
    return x_prev


@torch.no_grad()
def v_param_sample_loop(model, schedule, shape, device, steps=50, eta=0.0):
    """
    Generated samples using DDIM strided sampling.
    steps: Number of actual inference steps (e.g., 50)
    """
    # Create a subsequence of timesteps (e.g., 0, 20, 40... 980)
    # We flip it to go 980 -> 0 for generation
    # Note: We use 'step' logic to ensure we cover the range evenly
    total_steps = schedule.timesteps # This should be 1000

    # Create a sequence, e.g., [0, 20, 40, ..., 980]
    times = torch.linspace(0, total_steps - 1, steps=steps).long()

    # Reverse it: [980, ..., 20, 0]
    times = times.flip(0)

    # Convert to a list of pairs: [(980, 960), (960, 940), ..., (20, 0), (0, -1)]
    # We append -1 to indicate the final step to pure image
    time_pairs = []
    for i in range(len(times) - 1):
        time_pairs.append((times[i].item(), times[i+1].item()))
    time_pairs.append((times[-1].item(), -1))

    img = torch.randn(shape, device=device)

    # Progress bar for sampling
    for t_curr, t_prev in tqdm(time_pairs, desc="DDIM Sampling"):
        img = v_param_sample(model, schedule, img, t_curr, t_prev, eta)
    return img


def v_param_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs'):
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

            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = v_param_loss(model, schedule, x, t)

            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

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
            'avg_epoch_loss': avg_loss
        }
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        model.eval()
        samples = v_param_sample_loop(model, schedule, (16,3,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
