from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from .utils.measure import compute_eigenvalues, EigenvectorCache
from .utils.visualization import save_lambda_max_history

print('torch:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device ->', device)

# Utilities: beta schedules and forward q_sample
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionSchedule:
    def __init__(self, timesteps=50, device='cpu'):
        self.device = device
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps).to(device)
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x_start + b * noise

# Minimal UNet-like model adapted for CIFAR-10 (3 channels)
class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128, max_t=50.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
        )

        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, base_ch)
        self.norm2 = nn.GroupNorm(8, base_ch*2)
        self.norm3 = nn.GroupNorm(8, base_ch*2)

        self.max_t = max_t

    def forward(self, x, t):
        t = t.float().unsqueeze(-1) / float(self.max_t)
        temb = self.time_proj(t)
        h1 = self.act(self.norm1(self.conv1(x)))
        h2 = F.avg_pool2d(h1, 2)
        h2 = self.act(self.norm2(self.conv2(h2)))
        h3 = F.avg_pool2d(h2, 2)
        h3 = self.act(self.norm3(self.conv3(h3)))
        te = self.time_mlp(temb).unsqueeze(-1).unsqueeze(-1)
        h3 = h3 + te
        u1 = self.act(self.deconv1(h3))
        u1 = u1 + h2
        u2 = self.act(self.deconv2(u1))
        u2 = u2 + h1
        out = self.out(u2)
        return out

# Loss (MSE on noise) and sampling helpers
def p_losses(model, schedule, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = schedule.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)


# DDIM sampling
@torch.no_grad()
def p_sample(model, schedule, x, t, eta=0.0):
    """
    DDIM single-step update from t -> t-1.
    eta=0.0 -> deterministic DDIM (no added noise).
    eta>0.0 -> adds controlled noise (stochastic DDIM).
    """
    # scalar tensors for the required alphas
    alpha_bar_t = schedule.alphas_cumprod[t]            # ᾱ_t
    alpha_bar_prev = schedule.alphas_cumprod_prev[t]    # ᾱ_{t-1}
    sqrt_alpha_bar_t = schedule.sqrt_alphas_cumprod[t]  # sqrt(ᾱ_t)
    sqrt_alpha_bar_prev = schedule.sqrt_alphas_cumprod_prev[t]  # sqrt(ᾱ_{t-1})
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t]  # sqrt(1-ᾱ_t)

    # model predicts noise (ε_t)
    predicted_noise = model(x, torch.tensor([t], device=x.device).repeat(x.shape[0]))

    # predict x0 (same as DDPM)
    x0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    x0_pred = x0_pred.clamp(-1., 1.)

    # epsilon from model prediction (this is ε_t in DDIM paper)
    eps_pred = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

    # compute sigma_t (controls stochasticity). DDIM paper choice:
    # sigma_t = eta * sqrt((1 - ᾱ_{t-1})/(1 - ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
    # if eta==0 => sigma_t == 0 (deterministic)
    # ensure all ops on tensors for device consistency
    if t == 0:
        # when t==0 alpha_bar_prev == 1 so sqrt_alpha_bar_prev * x0_pred == x0_pred
        return sqrt_alpha_bar_prev * x0_pred

    frac = (1. - alpha_bar_prev) / (1. - alpha_bar_t)
    sigma_t = eta * torch.sqrt(frac) * torch.sqrt(1. - (alpha_bar_t / alpha_bar_prev))

    # compute the non stochastic component
    # note: ensure inside sqrt is non-negative by construction for valid schedules
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
