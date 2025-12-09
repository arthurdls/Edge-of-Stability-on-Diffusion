import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import tqdm

from implementations import base_implementation as base
from implementations import edm_style_preconditioning as edm
from implementations import v_parametrization as vparam
from implementations import min_snr_reweighting as snr
from implementations import progressive_difficulty_curriculum as pd
from implementations import adaptive_sampling as adap_s
from implementations import stf_smoothing as stf

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    raise ImportError("Package 'torchmetrics' not found. Please run: pip install torchmetrics torch-fidelity")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
NUM_SAMPLES = 10000  # Standard FID uses 10k or 50k; use 2k for speed
IMG_SIZE = 32

MODEL_CHECKPOINTS = {
    'Base (DDIM)': './test_base_lr_1e-3/checkpoint_epoch_100.pt',
    'EDM':         './test_edm_lr_1e-3/checkpoint_epoch_100.pt',
    'V-Param':     './test_vparam_lr_1e-3/checkpoint_epoch_100.pt',
    'Min-SNR':     './test_min_snr_lr_1e-3/checkpoint_epoch_100.pt',
    'Curriculum':  './test_progressive_lr_1e-4/checkpoint_epoch_100.pt',
    'Adaptive':    './test_adaptive_sampling_lr_1e-3/checkpoint_epoch_100.pt',
    'STF':         './test_stf_lr_1e-3/checkpoint_epoch_100.pt',
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: Ensures exact reproducibility but may slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_real_images_uint8(num_samples):
    """
    Loads real CIFAR-10 images and converts them to uint8 [0, 255]
    as required by torchmetrics.
    """
    print("Loading real images...")
    # Note: We load raw tensors [0, 1] then convert to uint8 [0, 255]
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_imgs = []
    count = 0
    for x, _ in loader:
        # x is [0, 1] float
        x = (x * 255).to(torch.uint8)
        all_imgs.append(x)
        count += x.shape[0]
        if count >= num_samples:
            break

    return torch.cat(all_imgs, dim=0)[:num_samples]

def load_model_and_generate(name, path, num_samples):
    """
    Loads model and generates samples. Returns uint8 tensor [N, 3, 32, 32].
    """
    print(f"\n--- Generating {num_samples} samples for {name} ---")

    # Common settings
    unet_kwargs = {'in_ch':3, 'base_ch':64, 'time_emb_dim':64}
    schedule_kwargs = {'timesteps': 1000, 'device': DEVICE}

    # 1. Initialize correct architecture
    if 'EDM' in name:
        unet = edm.UNet(**unet_kwargs)
        model = edm.EDMPrecond(unet).to(DEVICE)
        schedule = None
    elif 'V-Param' in name:
        model = vparam.UNet(**unet_kwargs).to(DEVICE)
        schedule = vparam.DiffusionSchedule(**schedule_kwargs)
    elif 'Min-SNR' in name:
        model = snr.UNet(**unet_kwargs).to(DEVICE)
        schedule = snr.MinSNRDiffusionSchedule(**schedule_kwargs)
    elif 'Adaptive' in name:
         model = adap_s.UNet(**unet_kwargs).to(DEVICE)
         schedule = adap_s.DiffusionSchedule(**schedule_kwargs)
    elif 'Curriculum' in name:
         model = pd.UNet(**unet_kwargs).to(DEVICE)
         schedule = pd.DiffusionSchedule(**schedule_kwargs)
    elif 'STF' in name:
         model = stf.UNet(**unet_kwargs).to(DEVICE)
         schedule = stf.DiffusionSchedule(**schedule_kwargs)
    else:
        model = base.UNet(**unet_kwargs).to(DEVICE)
        schedule = base.DiffusionSchedule(**schedule_kwargs)

    # 2. Load Weights
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found at {path}. Skipping.")
        return None

    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt

    # Strip '_orig_mod.' prefix if model was compiled
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Generate
    all_samples = []
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            current_bs = min(BATCH_SIZE, num_samples - len(all_samples) * BATCH_SIZE)

            if current_bs <= 0: break

            if 'EDM' in name:
                latents = torch.randn((current_bs, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
                samples = edm.edm_sample_loop(model, latents, num_steps=100)
            elif 'V-Param' in name:
                samples = vparam.v_param_sample_loop(model, schedule, (current_bs, 3, IMG_SIZE, IMG_SIZE), DEVICE, steps=100)
            else:
                samples = base.sample_loop(model, schedule, (current_bs, 3, IMG_SIZE, IMG_SIZE), DEVICE, steps=100)

            # Convert [-1, 1] float -> [0, 255] uint8 for FID
            samples = (samples.clamp(-1, 1) + 1) / 2.0
            samples = (samples * 255).to(torch.uint8)
            all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)

def feed_fid_in_batches(fid_metric, images_uint8, is_real, batch_size=128):
    """
    Feeds images to the FID metric in chunks to avoid OOM.
    """
    total = len(images_uint8)
    for i in range(0, total, batch_size):
        batch = images_uint8[i : i + batch_size].to(DEVICE)
        fid_metric.update(batch, real=is_real)

if __name__ == "__main__":
    seed_everything()

    # Initialize Metric (downloads Inception weights on first run)
    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)

    # 1. Get Real Images
    real_imgs = get_real_images_uint8(NUM_SAMPLES)

    results = {}

    for name, path in MODEL_CHECKPOINTS.items():
        seed_everything()

        # Generate Fakes
        fake_imgs = load_model_and_generate(name, path, NUM_SAMPLES)

        if fake_imgs is None: continue

        print(f"Calculating FID for {name}...")

        # Reset metric to clear old buffers
        fid.reset()

        # Feed Data (move to GPU in chunks if OOM occurs, otherwise full batch is faster)
        print("Feeding Real images to Inception...")
        feed_fid_in_batches(fid, real_imgs, is_real=True, batch_size=BATCH_SIZE)

        print("Feeding Fake images to Inception...")
        feed_fid_in_batches(fid, fake_imgs, is_real=False, batch_size=BATCH_SIZE)

        score = fid.compute()
        results[name] = score.item()

        print(f"{name} FID: {score.item():.4f}")

    # --- Plotting ---
    print("\n--- Plotting Results ---")
    names = list(results.keys())
    scores = list(results.values())

    # Sort
    sorted_pairs = sorted(zip(names, scores), key=lambda x: x[1])
    names, scores = zip(*sorted_pairs)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color='skyblue', edgecolor='black', width=0.6)

    plt.title(f'FID Scores (Lower is Better) - {NUM_SAMPLES} Samples', fontsize=14)
    plt.ylabel('FID Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fid_scores_torchmetrics.png', dpi=300)
    print("Saved chart to 'fid_scores_torchmetrics.png'")
    plt.show()