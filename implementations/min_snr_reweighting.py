"""
Implementation 3

Min-SNR Weighting Strategy Implementation
Reference: Hang et al., 2024

Summary of Changes from Standard DDPM/DDIM:

1.  Modified Loss Weighting ($w_t$):
    Treats the diffusion training process as a multi-task learning problem to address conflicting optimization directions between timesteps.
    Replaces the standard implicit weighting with a clamped Signal-to-Noise Ratio (SNR) strategy.
    The global loss weight for a timestep is defined as $w_t = min(SNR(t), gamma)$, effectively balancing the conflicts among different noise levels.

2.  Adaptation for Noise Prediction ($epsilon$):
    Standard DDPM training on $epsilon$ is mathematically equivalent to training on $x_0$ with an implicit weight of $SNR(t)$.
    To achieve the Min-SNR objective while keeping the $epsilon$-prediction target, the loss must be re-weighted by a factor of $min(1, gamma/SNR(t))$.
    This effectively clamps the loss contribution of high-SNR (low noise) steps, which otherwise dominate the gradients and slow down convergence.

3.  Hyperparameter ($gamma$):
    Introduces a truncation value $gamma$ to serve as the upper bound for the weights.
    The paper establishes $gamma=5$ as the robust default setting that works across different resolutions and architectures.

4.  Optimization Dynamics:
    Achieves results close to Pareto optimality (the theoretical ideal for multi-task learning) without the computational cost or instability of run-time adaptive weighting (like GradNorm).
    Significantly accelerates convergence, claiming a 3.4x speedup compared to previous weighting strategies.

https://arxiv.org/abs/2303.09556
"""
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
torch.backends.cudnn.benchmark = True

from implementations.base_implementation import (
    UNet, sample_loop, cosine_beta_schedule)

from torch.nn.attention import SDPBackend, sdpa_kernel
from implementations.utils.AEoSAnalyzer import AEoSAnalyzer

class MinSNRDiffusionSchedule:
    def __init__(self, timesteps=1000, device='cpu'):
        self.device = device
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # SNR = alpha_bar / (1 - alpha_bar)
        # We clamp the denominator to avoid division by zero at t=0 where alpha_bar ~ 1
        self.snr = alphas_cumprod / (1. - alphas_cumprod).clamp(min=1e-5)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x_start + b * noise


def min_snr_loss(model, schedule, x_start, t, snr_gamma=5.0):
    noise = torch.randn_like(x_start)
    x_noisy = schedule.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)

    # Calculate MSE per sample
    # Shape: [B, C, H, W] -> [B]
    loss = F.mse_loss(predicted_noise, noise, reduction='none')
    loss = loss.mean(dim=[1, 2, 3]) 

    # Get SNR for the current batch timesteps
    snr = schedule.snr[t]

    # Calculate weights: w = min(1, gamma / SNR)
    # Note: The paper defines w_t = min(SNR, gamma) for the general case.
    # For epsilon-prediction specifically, this simplifies to multiplying the 
    # standard loss by min(1, gamma / SNR).
    mse_loss_weights = torch.clamp(snr_gamma / snr, max=1.0)

    # Apply weights and average over batch
    return (loss * mse_loss_weights).mean()


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
            loss_measure = min_snr_loss(model, schedule, x_measure, t_measure)

            # 3. Analyze
            lmax = analyzer.log_step(
                model, opt, loss_measure, global_step, lr,
                timestep_label=ts_label, average_loss=current_avg_loss
            )

            print(f"  [{ts_label}] Lambda Max (P^-1 H): {lmax:.4f}")
            del loss_measure

    print(f"  Stability Threshold (38/Eta): {38.0/lr:.4f}")
    torch.cuda.empty_cache()


def min_snr_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', measure_bs=16, checkpoint=None):
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
            t = torch.randint(0, schedule.timesteps, (b,), device=device).long()

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
                loss = min_snr_loss(compiled_model, schedule, x, t)

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
        samples = sample_loop(compiled_model, schedule, (16,model.in_ch,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
