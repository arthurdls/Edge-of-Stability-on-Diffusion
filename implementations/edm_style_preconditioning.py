"""
Implementation 1

EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Implementation
Reference: Karras et al., 2022

Summary of Changes from Standard DDPM/DDIM:

1.  Continuous Noise Levels (Sigma):
    - Replaces discrete integer timesteps ($t$) with continuous noise levels ($sigma$).
    - $sigma$ represents the standard deviation of the noise added to the data.

2.  Preconditioning (The Wrapper):
    - The neural network $F_theta$ is not trained directly. Instead, it is wrapped
      to form a denoiser $D_theta(x; sigma)$.
    - Formula: $D_theta(x; sigma) = c_{skip}(sigma)x + c_{out}(sigma)F_theta(c_{in}(sigma)x; c_{noise}(sigma))$.
    - This ensures the network inputs and training targets always have unit variance,
      [cite_start]improving training stability[cite: 353].

3.  Training Objective:
    - The model predicts the clean image ($x_0$) rather than the noise ($epsilon$).
    - Training noise levels ($sigma$) are drawn from a Log-Normal distribution
      to focus training on the most perceptually relevant noise ranges.
    - Loss is weighted by $lambda(sigma)$ to balance contributions across noise levels.

4.  Deterministic Sampling:
    - Uses Heun's 2nd Order ODE solver instead of Euler/DDIM.
    - Reduces the number of steps required for high-quality generation (e.g., 18-35 steps).
    - Uses a specific time schedule where $sigma(t) = t$.

https://arxiv.org/abs/2206.00364
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
    UNet)


class EDMPrecond(nn.Module):
    def __init__(self, model, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        # Preconditioning formulas
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4.0

        # Model evaluation
        # F_theta(c_in * x, c_noise)
        F_x = self.model(c_in * x, c_noise.flatten())

        # D_theta(x, sigma) = c_skip * x + c_out * F_x
        D_x = c_skip * x + c_out * F_x

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_loss(model, x_start, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """
    Loss function.
    """
    # 1. Sample sigma from Log-Normal distribution
    rnd_normal = torch.randn([x_start.shape[0], 1, 1, 1], device=x_start.device)
    sigma = (rnd_normal * P_std + P_mean).exp()

    # 2. Calculate loss weight lambda(sigma)
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2

    # 3. Add noise
    noise = torch.randn_like(x_start)
    x_noisy = x_start + sigma * noise

    # 4. Run the preconditioned model D_theta
    # Note: model here should be the EDMPrecond wrapper
    D_x = model(x_noisy, sigma.flatten())

    # 5. Weighted MSE against the CLEAN image (x_start)
    loss = weight * ((D_x - x_start) ** 2)
    return loss.mean()


@torch.no_grad()
def edm_sample_loop(model, latents, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    """
    Deterministic sampling using Heun's 2nd order method
    """
    # 1. Define time steps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # 2. Initialize x_0 ~ N(0, sigma_max^2 * I)
    x_next = latents * sigma_max

    # 3. Loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # Euler step
        d_cur = (x_next - model(x_next, t_cur)) / t_cur
        x_suggest = x_next + (t_next - t_cur) * d_cur

        # Heun Correction
        if t_next > 0:
            d_suggest = (x_suggest - model(x_suggest, t_next)) / t_next
            x_next = x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_suggest)
        else:
            x_next = x_suggest

    return x_next


def edm_train_ddim(model, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', ema_decay=0.995):
    """
    Training loop for diffusion model.

    Args:
        model: The diffusion model (e.g., TinyUNet)
        train_loader: DataLoader for training data
        device: torch.device
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints and samples
        ema_decay: EMA decay factor (None to disable EMA)
    """
    unet = model.to(device)
    model = EDMPrecond(unet).to(device)

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
            # t = torch.randint(0, schedule.timesteps, (b,), device=device).long()

            opt.zero_grad()

            # --- 2. Autocast Context ---
            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = edm_loss(model, x) # No 'schedule' or 't' needed anymore

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
            'epoch': epoch+1
        }
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        ema_model.eval()
        # Generate initial noise with unit variance
        latents = torch.randn((16, 3, 32, 32), device=device)
        samples = edm_sample_loop(ema_model, latents) # Use new sampler

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
