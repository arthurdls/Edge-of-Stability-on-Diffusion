from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
torch.backends.cudnn.benchmark = True

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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Create a buffer for the frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        # t: [B] tensor of timesteps
        t_scaled = t.float()
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

# Multi-head flash attention block for the bottleneck.
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)

        # QKV projection
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Norm and reshape
        # Output: [B, H*W, C]
        x_norm = self.norm(x).flatten(2).transpose(1, 2) 

        # 2. QKV Projection
        # [B, Seq, 3*C] -> [B, Seq, 3, Heads, Head_Dim]
        qkv = self.qkv(x_norm).reshape(b, h*w, 3, self.num_heads, self.head_dim)

        # 3. Permute to [B, Heads, Seq, Head_Dim] for Flash Attention
        # Dimensions: 2=qkv, 0=batch, 3=heads, 1=seq, 4=head_dim
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # 4. Flash Attention (The Magic)
        # PyTorch automatically selects the fastest kernel (FlashAttn v2 or Memory Efficient)
        x_out = F.scaled_dot_product_attention(q, k, v)

        # 5. Reshape back
        # [B, Heads, Seq, Head_Dim] -> [B, Seq, C]
        x_out = x_out.transpose(1, 2).reshape(b, h*w, c)

        # 6. Output Projection & Residual
        x_out = self.proj(x_out)

        # Reshape back to spatial [B, C, H, W]
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w) 
        return x + x_out


class UNet(nn.Module):
    """
    A more robust UNet architecture incorporating:
    1. Sinusoidal Time Embeddings
    2. ResNet Blocks with Time Conditioning
    3. Self-Attention at the bottleneck
    4. Proper downsampling/upsampling
    """
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=64):
        super().__init__()

        # --- 1. Time Embedding ---
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
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
        self.down2_attn = SelfAttentionBlock(base_ch * 2)

        # 16x16 -> 8x8
        self.down3_downsample = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.down3_res1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3_res2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # --- 3. Bottleneck ---
        # 8x8
        self.mid_res1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.mid_attn = SelfAttentionBlock(base_ch * 4)
        self.mid_res2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # --- 4. Up-path ---
        # 8x8 -> 16x16
        self.up1_deconv = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up1_res1 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim) # (skip + input)
        self.up1_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.up1_attn = SelfAttentionBlock(base_ch * 2)

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
        h2 = self.down2_attn(h2)

        # --- Block 3 (8x8) ---
        h_down3 = self.down3_downsample(h2)
        h3 = self.down3_res1(h_down3, temb)
        h3 = self.down3_res2(h3, temb)

        # 3. Bottleneck
        h_mid = self.mid_res1(h3, temb)
        h_mid = self.mid_attn(h_mid) # Attention block doesn't need temb
        h_mid = self.mid_res2(h_mid, temb)

        # 4. Up-path
        # --- Block Up 1 (16x16) ---
        u1 = self.up1_deconv(h_mid)
        # Note the skip connection from h2
        u1_skip = torch.cat([u1, h2], dim=1) # [B, base_ch*2 + base_ch*2, 16, 16]
        u1_out = self.up1_res1(u1_skip, temb)
        u1_out = self.up1_res2(u1_out, temb)
        u1_out = self.up1_attn(u1_out)

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
def p_sample(model, schedule, x, t, t_prev, eta=0.0):
    """
    t: current timestep (e.g., 999)
    t_prev: next timestep (e.g., 979). If < 0, implies t_prev=-1 (final step)
    """
    # 1. Get alpha_bar for the CURRENT step (t)
    alpha_bar_t = schedule.alphas_cumprod[t]

    # 2. Get alpha_bar for the PREVIOUS step (t_prev) dynamically
    # If t_prev < 0, we are at the very end, so alpha_bar_prev is 1.0 (no noise)
    alpha_bar_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)

    # 3. Compute derived variables
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)

    # 4. Model Prediction
    # We need to expand t to a batch tensor
    t_tensor = torch.tensor([t], device=x.device).repeat(x.shape[0])
    predicted_noise = model(x, t_tensor)

    # 5. Predict x0 (Standard DDPM eq)
    x0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    x0_pred = x0_pred.clamp(-1., 1.) # Note that this introduces a non-linearity that isn't present in training

    # 6. Direction pointing to x_t (this is the "predicted noise" scaled)
    # Note: DDIM paper allows different "sigma" (eta).
    # Sigma calculation for DDIM
    # If eta=0 (Deterministic), sigma=0.
    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))

    # The direction component (this replaces the random noise in DDPM)
    pred_dir_xt = torch.sqrt(torch.clamp(1. - alpha_bar_prev - sigma_t**2, min=0.0)) * predicted_noise

    # 7. Compute x_{t-1} (or x_{t_prev})
    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + pred_dir_xt

    # Add noise only if eta > 0 (Stochastic DDIM)
    if eta > 0:
        noise = torch.randn_like(x)
        x_prev = x_prev + sigma_t * noise

    return x_prev


@torch.no_grad()
def sample_loop(model, schedule, shape, device, steps=50, eta=0.0):
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
        img = p_sample(model, schedule, img, t_curr, t_prev, eta)
    return img



def train_ddim(model, schedule, train_loader, device, epochs=100, lr=2e-4, save_dir='./runs', ema_decay=0.995):
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
                loss = p_losses(model, schedule, x, t)

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
