import math
import os
from pathlib import Path
import torch

# Utilities: beta schedules and forward q_sample
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionSchedule:
    def __init__(self, timesteps=1000, device='cpu'):
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
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128):
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

    def forward(self, x, t):
        t = t.float().unsqueeze(-1) / float(1000)
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

@torch.no_grad()
def p_sample(model, schedule, x, t):
    # x: (B,C,H,W), t: int scalar
    betas_t = schedule.betas[t]
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t]
    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t]
    predicted_noise = model(x, torch.tensor([t], device=x.device).repeat(x.shape[0]))
    x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
    x0_pred = x0_pred.clamp(-1.,1.)
    posterior_mean = (
        schedule.alphas[t].sqrt() * x0_pred +
        (1 - schedule.alphas[t]).sqrt() * predicted_noise
    )
    if t == 0:
        return posterior_mean
    else:
        var = schedule.posterior_variance[t]
        noise = torch.randn_like(x)
        return posterior_mean + var.sqrt() * noise

@torch.no_grad()
def sample_loop(model, schedule, shape, device):
    img = torch.randn(shape, device=device)
    for t in reversed(range(schedule.timesteps)):
        img = p_sample(model, schedule, img, t)
    return img

# Training loop using SGD (with momentum)
def train_ddpm(model, schedule, train_loader, device, epochs=20, lr=1e-2, save_dir='./runs', ema_decay=0.995):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # EMA params
    ema_params = {n: p.detach().clone() for n,p in model.named_parameters()}
    use_ema = ema_decay is not None

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
            if global_step % 200 == 0:
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
        utils.save_image(grid, save_dir / f'samples_epoch_{epoch+1}.png', nrow=4)
        print('Saved samples to', save_dir / f'samples_epoch_{epoch+1}.png')

        # restore original weights
        if use_ema:
            for n,p in model.named_parameters():
                p.data.copy_(backup[n].to(p.device))
