"""
Implementation 1

The original code trains the model to predict noise (ε).
The EDM formulation trains the model (which it calls D_theta)
    to be part of a larger function F_theta that predicts the clean image (x_0).

This "preconditioning" is just a wrapper around the model that scales its inputs and outputs.
"""

from implementations.base_implementation import (
    DiffusionSchedule, TinyUNet)

from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import utils
from .utils.measure import compute_eigenvalues, EigenvectorCache
from .utils.visualization import save_lambda_max_history

print('torch:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device ->', device)

# Loss (MSE on x0, with EDM preconditioning)
def p_losses(model, schedule, x_start, t):
    """
    EDM-style preconditioned loss.
    The model D_θ predicts the denoised component, and the
    full preconditioned function F_θ predicts x_0.
    """
    noise = torch.randn_like(x_start)
    # x_t (noisy image)
    x_noisy = schedule.q_sample(x_start=x_start, t=t, noise=noise)

    # Get EDM scalers for the batch
    # We use .to(t.device) to ensure scalers are on the same device as t
    a = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(t.device)
    b = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(t.device)
    alpha_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1).to(t.device)

    # c_in = a
    # c_skip = alpha_bar
    # c_out = b
    # w = 1.0 / alpha_bar (loss weight)
    w = (1.0 / alpha_bar).view(-1, 1, 1, 1)

    # EDM preconditioned forward pass:
    # F_theta = c_skip * x_t + c_out * D(c_in * x_t, t)
    network_input = a * x_noisy
    network_output = model(network_input, t)
    F_theta = alpha_bar * x_noisy + b * network_output

    # Target is x_0 (x_start)
    target = x_start

    # Weighted MSE loss
    loss_per_pixel = F.mse_loss(F_theta, target, reduction='none')
    weighted_loss = w * loss_per_pixel
    return weighted_loss.mean()


# DDIM sampling (using EDM-preconditioned model)
@torch.no_grad()
def p_sample(model, schedule, x, t, eta=0.0):
    """
    DDIM single-step update from t -> t-1.
    Uses the EDM-preconditioned model F_theta to predict x0 directly.
    """
    # scalar tensors for the required alphas
    alpha_bar_t = schedule.alphas_cumprod[t]           # ᾱ_t
    alpha_bar_prev = schedule.alphas_cumprod_prev[t]   # ᾱ_{t-1}
    sqrt_alpha_bar_t = schedule.sqrt_alphas_cumprod[t] # sqrt(ᾱ_t)
    sqrt_alpha_bar_prev = schedule.sqrt_alphas_cumprod_prev[t] # sqrt(ᾱ_{t-1})
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t] # sqrt(1-ᾱ_t)

    # Get EDM scalers for this single timestep t
    # Need to move scalars to the device of x for tensor ops
    a = schedule.sqrt_alphas_cumprod[t].to(x.device)
    b = schedule.sqrt_one_minus_alphas_cumprod[t].to(x.device)
    alpha_bar = schedule.alphas_cumprod[t].to(x.device)
    
    t_tensor = torch.tensor([t], device=x.device).repeat(x.shape[0])

    # --- MODIFIED PART ---
    # model now predicts x0 (F_theta) directly via preconditioning
    # F_theta = c_skip * x_t + c_out * D(c_in * x_t, t)
    network_input = a * x
    network_output = model(network_input, t_tensor)
    x0_pred = alpha_bar * x + b * network_output
    # --- END MODIFIED PART ---

    # clamp x0 prediction
    x0_pred = x0_pred.clamp(-1., 1.)

    # epsilon from model prediction (this is ε_t in DDIM paper)
    eps_pred = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

    # compute sigma_t (controls stochasticity).
    if t == 0:
        return sqrt_alpha_bar_prev * x0_pred

    frac = (1. - alpha_bar_prev) / (1. - alpha_bar_t)
    sigma_t = eta * torch.sqrt(frac) * torch.sqrt(1. - (alpha_bar_t / alpha_bar_prev))

    # compute the non stochastic component
    non_stochastic_coeff = torch.sqrt(torch.clamp(1. - alpha_bar_prev - sigma_t * sigma_t, min=0.0))

    x_prev = sqrt_alpha_bar_prev * x0_pred + non_stochastic_coeff * eps_pred

    if eta > 0.0:
        noise = torch.randn_like(x)
        x_prev = x_prev + sigma_t * noise

    return x_prev


@torch.no_grad()
def sample_loop(model, schedule, shape, device, eta=0.0):
    img = torch.randn(shape, device=device)
    for t in reversed(range(schedule.timesteps)):
        img = p_sample(model, schedule, img, t, eta)
    return img


# Training loop using SGD (with momentum)
def train_ddim(model, schedule, train_loader, device, epochs=20, lr=1e-2, save_dir='./runs', ema_decay=0.995,
               compute_lambdamax=False, lambdamax_freq=500, num_eigenvalues=1, use_power_iteration=False):
    """
    Training loop for diffusion model with optional Hessian eigenvalue computation.
    
    Args:
        model: The diffusion model (e.g., TinyUNet)
        schedule: DiffusionSchedule instance
        train_loader: DataLoader for training data
        device: torch.device
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints and samples
        ema_decay: EMA decay factor (None to disable EMA)
        compute_lambdamax: If True, compute largest Hessian eigenvalue during training
        lambdamax_freq: Frequency (in steps) to compute lambda max
        num_eigenvalues: Number of eigenvalues to compute (default: 1 for lambda max)
        use_power_iteration: If True, use power iteration instead of LOBPCG
    """
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # EMA params
    ema_params = {n: p.detach().clone() for n,p in model.named_parameters()}
    use_ema = ema_decay is not None

    # Eigenvector cache for warm starts in Hessian computation
    eigenvector_cache = EigenvectorCache(max_eigenvectors=num_eigenvalues) if compute_lambdamax else None

    # Store lambda max values for visualization
    lambda_max_history = [] if compute_lambdamax else None

    global_step = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        running_loss = 0.0
        for i, (x, _) in enumerate(pbar):
            x = x.to(device)
            b = x.shape[0]
            t = torch.randint(0, schedule.timesteps, (b,), device=device).long()
            loss = p_losses(model, schedule, x, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # EMA update
            if use_ema:
                with torch.no_grad():
                    for n,p in model.named_parameters():
                        ema_params[n].mul_(ema_decay).add_(p.detach(), alpha=1-ema_decay)

            running_loss += loss.item()
            global_step += 1
            
            # Compute Hessian largest eigenvalue (lambda max) if requested
            if compute_lambdamax and global_step % lambdamax_freq == 0:
                # Temporarily set model to eval mode and ensure gradients are enabled
                model.eval()
                opt.zero_grad()
                
                # Ensure model parameters require gradients for Hessian computation
                for param in model.parameters():
                    param.requires_grad = True
                
                # Use a batch from the dataset for Hessian computation
                with torch.enable_grad():
                    # Compute loss on batch with gradient computation enabled
                    x_batch = x.detach().requires_grad_(False)
                    b_batch = x_batch.shape[0]
                    t_batch = torch.randint(0, schedule.timesteps, (b_batch,), device=device).long()
                    
                    # Compute loss - the loss tensor needs to retain computational graph
                    loss_for_hessian = p_losses(model, schedule, x_batch, t_batch)
                    
                    # Compute eigenvalues (largest eigenvalue = lambda max)
                    try:
                        eigenvals = compute_eigenvalues(
                            loss_for_hessian,
                            model,
                            k=num_eigenvalues,
                            max_iterations=100 if not use_power_iteration else 1000,
                            reltol=1e-2 if num_eigenvalues < 6 else 0.03,
                            eigenvector_cache=eigenvector_cache,
                            return_eigenvectors=False,
                            use_power_iteration=use_power_iteration
                        )
                        
                        if num_eigenvalues == 1:
                            lambda_max = eigenvals.item()
                        else:
                            lambda_max = eigenvals[0].item()
                        
                        # Store lambda max value for visualization
                        if lambda_max_history is not None:
                            lambda_max_history.append({
                                'step': global_step,
                                'epoch': epoch + 1,
                                'lambda_max': lambda_max,
                                'loss': loss.item()
                            })
                        
                        print(f'Step {global_step}: Lambda Max = {lambda_max:.6f}')
                        
                        # Update progress bar with lambda max
                        if global_step % 200 == 0:
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'lr': f'{scheduler.get_last_lr()[0]:.5f}',
                                'λ_max': f'{lambda_max:.4f}'
                            })
                    except Exception as e:
                        print(f'Warning: Failed to compute lambda max at step {global_step}: {e}')
                
                model.train()
            
            if global_step % 200 == 0 and not (compute_lambdamax and global_step % lambdamax_freq == 0):
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.5f}'})

        avg_loss = running_loss / len(train_loader)
        print(f'End epoch {epoch+1}, avg loss {avg_loss:.4f}')

        # save checkpoint (model + ema)
        ckpt = {'model_state': model.state_dict(), 'ema_state': {k:v.cpu() for k,v in ema_params.items()}, 'epoch': epoch+1}
        torch.save(ckpt, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # sample and save a grid using EMA weights for nicer samples
        model.eval()
        # swap to ema weights for sampling
        if use_ema:
            backup = {n: p.detach().clone() for n,p in model.named_parameters()}
            for n,p in model.named_parameters():
                p.data.copy_(ema_params[n].to(p.device))

        samples = sample_loop(model, schedule, (16,3,32,32), device=device)
        grid = (samples.clamp(-1,1) + 1) / 2.0  # to [0,1]
        utils.save_image(grid.cpu(), save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')

        # restore original weights
        if use_ema:
            for n,p in model.named_parameters():
                p.data.copy_(backup[n].to(p.device))
    
    # Save lambda max history if computed
    if lambda_max_history is not None and len(lambda_max_history) > 0:
        lambda_max_csv_path = save_dir / 'lambda_max_history.csv'
        save_lambda_max_history(lambda_max_history, str(lambda_max_csv_path))
        print(f'Saved lambda max history to {lambda_max_csv_path}')
    
    # Return lambda max history if computed
    return lambda_max_history
