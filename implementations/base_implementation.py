from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torchvision import utils
from .utils.measure import compute_eigenvalues, EigenvectorCache
from .utils.visualization import save_lambda_max_history

print('torch:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device ->', device)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as described in "Improved Denoising Diffusion Probabilistic Models".
    (https://arxiv.org/abs/2102.09672)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f_t = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class DiffusionSchedule:
    def __init__(self, timesteps=50, device='cpu'):
        self.device = device
        self.timesteps = timesteps
        # betas = linear_beta_schedule(timesteps).to(device)
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x_start + b * noise

# This is the standard, superior method for time encoding.
class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard Sinusoidal Time Embedding, as used in "Attention Is All You Need"
    and subsequent diffusion models.
    """
    def __init__(self, dim, max_t=1000.0):
        super().__init__()
        self.dim = dim
        self.max_t = max_t
        
        # Create a buffer for the frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        # t: [B] tensor of timesteps
        t_scaled = t.float() * (1000.0 / self.max_t) # Scale timesteps to a standard range
        pos_emb = t_scaled.unsqueeze(-1) * self.inv_freq.to(t.device)
        emb = torch.cat([pos_emb.sin(), pos_emb.cos()], dim=-1) # [B, dim]
        return emb

# This block adds residual connections and handles time conditioning.
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.act = nn.SiLU()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            self.act,
            nn.Linear(time_emb_dim, out_ch * 2) # Project to gain and bias
        )
        
        # First convolution block (pre-activation)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Second convolution block (pre-activation)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temb):
        # --- Main Path ---
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        
        # --- Time Conditioning ---
        # Project time embedding and reshape to (B, C*2, 1, 1)
        time_bias = self.time_mlp(temb).unsqueeze(-1).unsqueeze(-1)
        # Split into scale and shift (B, C, 1, 1)
        scale, shift = time_bias.chunk(2, dim=1)
        
        # Apply time conditioning (AdaGN-like)
        h = self.act(self.norm2(h))
        h = h * (1 + scale) + shift # Modulate the normalized features
        
        h = self.dropout(h)
        h = self.conv2(h)
        
        # --- Add Residual ---
        return h + self.shortcut(x)

# Standard multi-head attention block for the bottleneck.
# class SelfAttentionBlock(nn.Module):
#     def __init__(self, channels, num_heads=4):
#         super().__init__()
#         self.channels = channels
#         self.num_heads = num_heads
        
#         self.norm = nn.GroupNorm(8, channels)
#         # Use built-in MultiheadAttention
#         self.attn = MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         b, c, h, w = x.shape
        
#         # 1. Normalize and flatten
#         h_norm = self.norm(x)
#         h_flat = h_norm.reshape(b, c, h * w).permute(0, 2, 1) # [B, H*W, C]
        
#         # 2. Apply self-attention
#         # Query, Key, and Value are all from the same input
#         attn_output, _ = self.attn(h_flat, h_flat, h_flat) # [B, H*W, C]
        
#         # 3. Reshape and add residual
#         attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w) # [B, C, H, W]
        
#         return x + attn_output


class UNet(nn.Module):
    """
    A more robust UNet architecture incorporating:
    1. Sinusoidal Time Embeddings
    2. ResNet Blocks with Time Conditioning
    3. Self-Attention at the bottleneck
    4. Proper downsampling/upsampling
    """
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=128, max_t=1000.0):
        super().__init__()
        self.max_t = max_t
        
        # --- 1. Time Embedding ---
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim, max_t=max_t)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # --- 2. Down-path ---
        self.conv_in = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        
        # 32x32 -> 32x32
        self.down1_res1 = ResBlock(base_ch, base_ch, time_emb_dim)
        # self.down1_res2 = ResBlock(base_ch, base_ch, time_emb_dim)
        
        # 32x32 -> 16x16
        self.down2_downsample = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.down2_res1 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        
        # 16x16 -> 8x8
        self.down3_downsample = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.down3_res1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3_res2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # --- 3. Bottleneck ---
        # 8x8
        self.mid_res1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        # self.mid_attn = SelfAttentionBlock(base_ch * 4)
        self.mid_res2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # --- 4. Up-path ---
        # 8x8 -> 16x16
        self.up1_deconv = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up1_res1 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim) # (skip + input)
        self.up1_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        # 16x16 -> 32x32
        self.up2_deconv = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.up2_res1 = ResBlock(base_ch * 2, base_ch, time_emb_dim) # (skip + input)
        # self.up2_res2 = ResBlock(base_ch, base_ch, time_emb_dim)

        # --- 5. Output ---
        self.conv_out_norm = nn.GroupNorm(8, base_ch)
        self.conv_out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(base_ch, in_ch, kernel_size=1)
        
    def forward(self, x, t):
        # x: [B, 3, 32, 32]
        # t: [B]
        
        # 1. Get time embedding
        temb_sin = self.time_emb(t)
        temb = self.time_mlp(temb_sin) # [B, time_emb_dim]
        
        # 2. Down-path
        h_in = self.conv_in(x) # [B, base_ch, 32, 32]
        
        # --- Block 1 (32x32) ---
        h1 = self.down1_res1(h_in, temb)
        # h1 = self.down1_res2(h1, temb)
        
        # --- Block 2 (16x16) ---
        h_down2 = self.down2_downsample(h1)
        h2 = self.down2_res1(h_down2, temb)
        h2 = self.down2_res2(h2, temb)
        
        # --- Block 3 (8x8) ---
        h_down3 = self.down3_downsample(h2)
        h3 = self.down3_res1(h_down3, temb)
        h3 = self.down3_res2(h3, temb)
        
        # 3. Bottleneck
        h_mid = self.mid_res1(h3, temb)
        # h_mid = self.mid_attn(h_mid) # Attention block doesn't need temb
        h_mid = self.mid_res2(h_mid, temb)
        
        # 4. Up-path
        # --- Block Up 1 (16x16) ---
        u1 = self.up1_deconv(h_mid)
        # Note the skip connection from h2
        u1_skip = torch.cat([u1, h2], dim=1) # [B, base_ch*2 + base_ch*2, 16, 16]
        u1_out = self.up1_res1(u1_skip, temb)
        u1_out = self.up1_res2(u1_out, temb)
        
        # --- Block Up 2 (32x32) ---
        u2 = self.up2_deconv(u1_out)
        # Note the skip connection from h1
        u2_skip = torch.cat([u2, h1], dim=1) # [B, base_ch + base_ch, 32, 32]
        u2_out = self.up2_res1(u2_skip, temb)
        # u2_out = self.up2_res2(u2_out, temb)
        
        # 5. Output
        h_out = self.conv_out_act(self.conv_out_norm(u2_out))
        out = self.conv_out(h_out)
        
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
def train_ddim(model, schedule, train_loader, device, epochs=20, lr=2e-4, save_dir='./runs', ema_decay=0.995,
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
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_lr = 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader), eta_min=final_lr)
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
                        if global_step % 100 == 0:
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'lr': f'{scheduler.get_last_lr()[0]:.5f}',
                                'λ_max': f'{lambda_max:.4f}'
                            })
                    except Exception as e:
                        print(f'Warning: Failed to compute lambda max at step {global_step}: {e}')
                
                model.train()
            
            if global_step % 100 == 0 and not (compute_lambdamax and global_step % lambdamax_freq == 0):
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
