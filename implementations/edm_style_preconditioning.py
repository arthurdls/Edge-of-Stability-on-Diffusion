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
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    UNet)

from torch.nn.attention import SDPBackend, sdpa_kernel
from implementations.utils.AEoSAnalyzer import AEoSAnalyzer


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


@torch.compiler.disable
def measure_aeos_step(model, opt, analyzer, global_step, lr, monitor_timesteps, device, x, current_avg_loss, measure_bs):
    """
    Helper function to run the AEoS measurement in strict eager mode.
    """

    # We still need the math backend for higher-order derivatives
    with sdpa_kernel([SDPBackend.MATH]):
        for ts_req in monitor_timesteps:
            torch.cuda.empty_cache()

            # 1. Prepare Data
            x_measure = x[:measure_bs]
            ts_label = f"σ={ts_req}"

            # 2. Forward Pass (Eager Mode)
            loss_measure = edm_loss(model, x_measure) # No 'schedule' or 't' needed anymore

            # 3. Analyze
            lmax = analyzer.log_step(
                model, opt, loss_measure, global_step, lr,
                timestep_label=ts_label, average_loss=current_avg_loss
            )

            print(f"  [{ts_label}] Lambda Max (P^-1 H): {lmax:.4f}")
            del loss_measure

    print(f"  Stability Threshold (38/Eta): {38.0/lr:.4f}")
    torch.cuda.empty_cache()

def edm_train_ddim(model, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', measure_bs=16, checkpoint=None):
    """
    Training loop for diffusion model.

    Args:
        model: The diffusion model (e.g., TinyUNet)
        train_loader: DataLoader for training data
        device: torch.device
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints and samples
    """
    unet = model.to(device)
    model = EDMPrecond(unet).to(device)
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
    monitor_timesteps = torch.linspace(0.003, 79, 10)

    # Initialize starting epoch
    start_epoch = 0

    # --- Checkpoint Loading ---
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt_data = torch.load(checkpoint, map_location=device, weights_only=True)
        
        # Load state into the wrapped EDMPrecond model
        model.load_state_dict(ckpt_data['model_state'])
        
        # Load optimizer and scaler
        opt.load_state_dict(ckpt_data['optimizer_state'])
        scaler.load_state_dict(ckpt_data['scaler_state'])
        
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

            opt.zero_grad()

            # --- AEoS Measurement Check ---
            if global_step > 0 and global_step % 1 == 0:
                print(f"\n[Step {global_step}] Measuring AEoS...")

                current_avg_loss = interval_loss / 1
                interval_loss = 0.0 # reset every 1 interval

                measure_aeos_step(
                    model, opt, analyzer, global_step, lr,
                    monitor_timesteps, device, x,
                    current_avg_loss, measure_bs
                )

            # Runs the forward pass in FP16 (half precision) where safe, 
            # but keeps critical ops (like softmax or reductions) in FP32.
            with torch.amp.autocast('cuda'):
                loss = edm_loss(compiled_model, x) # No 'schedule' or 't' needed anymore

            # Scales loss to prevent underflow in FP16 gradients
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

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
            'avg_epoch_loss': avg_loss
        }
        # torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(ckpt, save_dir / f'latest.pt')

        compiled_model.eval()
        # Generate initial noise with unit variance
        latents = torch.randn((16, unet.in_ch, 32, 32), device=device)
        samples = edm_sample_loop(compiled_model, latents) # Use new sampler

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')