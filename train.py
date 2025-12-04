import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import argparse
import os

from implementations import base_implementation as base
from implementations import edm_style_preconditioning as edm
from implementations import v_parametrization as vparam
from implementations import min_snr_reweighting as snr
from implementations import progressive_difficulty_curriculum as pd
from implementations import adaptive_sampling as adap_s
from implementations import stf_smoothing as stf

import numpy as np
import random

print('torch:', torch.__version__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device ->', DEVICE)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: Ensures exact reproducibility but may slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


seed_everything()

def get_dataloaders(batch_size=128, img_size=32, num_workers=4, training_subset_size=None):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # map to [-1, 1] for CIFAR10
        # transforms.Normalize((0.5,), (0.5,)),  # map to [-1,1] for MNIST
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if training_subset_size:
        indices = list(range(training_subset_size))
        train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# Helper to show saved sample grid
from PIL import Image
def show_image(path, figsize=(6,6)):
    img = Image.open(path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

parser = argparse.ArgumentParser(description="Training Script for EoS on Diffusion")
parser.add_argument('--lr', type=str, default='', help='Learning rate (default: 1e-3), use 1e-3 format')
parser.add_argument('--base', action='store_true', help='Run BASE implementation')
parser.add_argument('--edm', action='store_true', help='Run EDM implementation')
parser.add_argument('--vparam', action='store_true', help='Run V-PARAM implementation')
parser.add_argument('--snr', action='store_true', help='Run MIN-SNR implementation')
parser.add_argument('--pd', action='store_true', help='Run Progressive Difficulty implementation')
parser.add_argument('--adap_s', action='store_true', help='Run Adaptive Sampling implementation')
parser.add_argument('--stf', action='store_true', help='Run STF Smoothing implementation')
args = parser.parse_args()

BASE = args.base
EDM = args.edm
VPARAM = args.vparam
SNR = args.snr
PD = args.pd
ADAP_S = args.adap_s
STF = args.stf

if args.lr == '':
    LR_dict = {
        1e-3: '_lr_1e-3',
        5e-4: '_lr_5e-4',
        1e-4: '_lr_1e-4',
        5e-5: '_lr_5e-5',
        1e-5: '_lr_1e-5',
    }
else:
    LR_dict = {
        float(args.lr): f'_lr_{args.lr}',
    }

CHECKPOINT_ROOT = "round1"
CHECKPOINT_FOLDER = []
for item in os.listdir(CHECKPOINT_ROOT):
    item_path = os.path.join(CHECKPOINT_ROOT, item)
    if os.path.isdir(item_path):
        CHECKPOINT_FOLDER.append(item_path)

def lookup_checkpoint(save_dir_name):
    for folder in CHECKPOINT_FOLDER:
        if save_dir_name in folder:
            return os.path.join(folder, "latest.pt")
    return None

TIMESTEPS = 1000
EPOCHS = 1000
BATCH_SIZE = 10_000
AEOS_MEASURE_BS = 1000
EMB_DIM_SIZE = 32

print("--- LOADING DATA ---")
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, img_size=32, num_workers=2, training_subset_size=BATCH_SIZE)

TOTAL_DATA = len(train_loader.dataset)
print(f"\nTotal training data: {TOTAL_DATA} images")

print("--- RUNNING IMPLEMENTATIONS ---")
for LR, TEST_NAME_ADD_ON in LR_dict.items():
    print("--- SETTINGS ---")
    print(f"LR: {LR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"AEoS Measure Batch Size: {AEOS_MEASURE_BS}")
    print(f"Embedding Dimension Size: {EMB_DIM_SIZE}")

    unet_setting = {
        'in_ch':3, # 1 for MNIST, 3 for CIFAR10
        'base_ch':EMB_DIM_SIZE, # 32 for MNIST and for CIFAR10
        'time_emb_dim':EMB_DIM_SIZE, # 32 for MNIST and for CIFAR10
    }
    schedule_settings = {
        'timesteps':TIMESTEPS,
        'device':DEVICE
    }
    train_settings = {
        'train_loader':train_loader,
        'device':DEVICE,
        'epochs':EPOCHS,
        'lr':LR,
        'measure_bs': AEOS_MEASURE_BS,
    }

    if BASE:
        print("Running BASE Implementation")
        seed_everything()
        model = base.UNet(**unet_setting)
        schedule = base.DiffusionSchedule(**schedule_settings)
        save_dir = './test_base' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        base.train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if EDM:
        print("Running EDM Implementation")
        seed_everything()
        model = edm.UNet(**unet_setting)
        save_dir = './test_edm_preconditioning' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        edm.edm_train_ddim(
            model=model,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if VPARAM:
        print("Running VPARAM Implementation")
        seed_everything()
        model = vparam.UNet(**unet_setting)
        schedule = vparam.DiffusionSchedule(**schedule_settings)
        save_dir = './test_v_parametrization' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        vparam.v_param_train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if SNR:
        print("Running SNR Implementation")
        seed_everything()
        model = snr.UNet(**unet_setting)
        schedule = snr.MinSNRDiffusionSchedule(**schedule_settings)
        save_dir = './test_min_snr_reweighting' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        snr.min_snr_train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if PD:
        print("Running PD Implementation")
        seed_everything()
        model = pd.UNet(**unet_setting)
        schedule = pd.DiffusionSchedule(**schedule_settings)
        save_dir = './test_progressive_difficulty' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        pd.progressive_train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if ADAP_S:
        print("Running ADAP_S Implementation")
        seed_everything()
        model = adap_s.UNet(**unet_setting)
        schedule = adap_s.DiffusionSchedule(**schedule_settings)
        save_dir = './test_adaptive_sampling' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        adap_s.adaptive_train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )

    if STF:
        print("Running STF Implementation")
        seed_everything()
        model = stf.UNet(**unet_setting)
        schedule = stf.DiffusionSchedule(**schedule_settings)
        save_dir = './test_stf_smoothing' + TEST_NAME_ADD_ON
        checkpoint = lookup_checkpoint(save_dir[2:])
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        stf.stf_train_ddim(
            model=model,
            schedule=schedule,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **train_settings
        )
    print(f"--- ALL IMPLEMENTATIONS FINISHED FOR LR = {LR} ----")
print("--- ALL IMPLEMENTATIONS FINISHED FOR ALL LR ----")