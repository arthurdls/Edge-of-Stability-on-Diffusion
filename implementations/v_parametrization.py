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
from torch.nn.attention import SDPBackend, sdpa_kernel
from implementations.utils.AEoSAnalyzer import AEoSAnalyzer

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
            loss_measure = v_param_loss(model, schedule, x_measure, t_measure)

            # 3. Analyze
            lmax = analyzer.log_step(
                model, opt, loss_measure, global_step, lr,
                timestep_label=ts_label, average_loss=current_avg_loss
            )

            print(f"  [{ts_label}] Lambda Max (P^-1 H): {lmax:.4f}")
            del loss_measure

    print(f"  Stability Threshold (38/Eta): {38.0/lr:.4f}")
    torch.cuda.empty_cache()



def v_param_train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', measure_bs=16, checkpoint=None):
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
                loss = v_param_loss(compiled_model, schedule, x, t)

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
        samples = v_param_sample_loop(compiled_model, schedule, (16,model.in_ch,32,32), device=device, steps=100)

        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')
