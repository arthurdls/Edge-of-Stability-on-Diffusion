# Edge of Stability on Diffusion Models

This repository contains the official implementation for the paper **"Edge of Stability in Diffusion Models"**. It provides a framework to investigate the training dynamics of diffusion models through the lens of the *Adaptive Edge of Stability (AEoS)*.

We provide a modular codebase comparing standard DDPM/DDIM training against six state-of-the-art stabilization techniques, all instrumented with exact Hessian spectral analysis to monitor loss landscape sharpness during training.

---

## Quick Start

**Setup** (virtual environment highly recommended):

Using Conda:
```bash
conda create -n eos-diffusion python=3.12
conda activate eos-diffusion
pip install -r requirements.txt
````

OR using venv:

```bash
python3 -m venv eos-diffusion
source eos-diffusion/bin/activate
pip install -r requirements.txt
```

## Usage

### Training from CLI

The main entry point is `train.py`. You can select specific diffusion implementations using flags and specify the learning rate.

**Example: Train Baseline DDIM**

```bash
python train.py --base --lr 1e-3
```

**Example: Train EDM and V-Parameterization variants**

```bash
python train.py --edm --vparam --lr 5e-4
```

**Available Flags:**

  * `--base`: Standard DDIM (Baseline)
  * `--edm`: EDM-Style Preconditioning
  * `--vparam`: V-Parameterization
  * `--snr`: Min-SNR Weighting
  * `--pd`: Progressive Difficulty Curriculum
  * `--adap_s`: Adaptive Sampling
  * `--stf`: Stable Target Field Smoothing
  * `--lr`: Learning rate (e.g., `1e-3`, `5e-4`, `1e-4`). Default runs a sweep of learning rates.

### Interactive Notebook

For interactive experimentation and visualization, use the provided Jupyter notebook:

```bash
jupyter notebook diffusion_training.ipynb
```

## Implementations

This repository includes PyTorch implementations of seven diffusion model variants, located in the `implementations/` directory:

1.  **Baseline DDIM** (`base_implementation.py`)
      * Standard $\epsilon$-prediction with uniform timestep sampling.
2.  **EDM Preconditioning** (`edm_style_preconditioning.py`)
      * Continuous noise levels with scale-equalized input/output weighting (Karras et al., 2022).
3.  **V-Parameterization** (`v_parametrization.py`)
      * Predicts velocity $v$ to avoid singularities at low SNR (Salimans & Ho, 2022).
4.  **Min-SNR Weighting** (`min_snr_reweighting.py`)
      * Clamps gradients from high-SNR regions to balance multi-task conflicts (Hang et al., 2024).
5.  **Progressive Difficulty** (`progressive_difficulty_curriculum.py`)
      * Curriculum learning that moves from easy (high noise) to hard (low noise) tasks (Kim et al., 2025).
6.  **Adaptive Sampling** (`adaptive_sampling.py`)
      * Learnable timestep sampler optimized via policy gradient (Kim et al., 2025).
7.  **Stable Target Field** (`stf_smoothing.py`)
      * Reduces target variance by regressing towards a batch-averaged reference (Xu et al., 2023).

## Spectral Analysis (AEoS)

The core analytical tool is located in `implementations/utils/`.

  * **`AEoSAnalyzer`**: Computes the largest eigenvalue ($\lambda_1$) of the preconditioned Hessian $P^{-1}H$ at every step.
  * **`lobpcg.py`**: Implements the Locally Optimal Block Preconditioned Conjugate Gradient algorithm for efficient exact eigenvalue computation without materializing the full Hessian [https://github.com/arseniqum/edge-of-stochastic-stability].

## Project Structure

```
├── implementations/          # Model variants and training logic
│   ├── utils/                # Spectral analysis and math utilities
│   ├── base_implementation.py
│   ├── edm_style_preconditioning.py
│   └── ... (other variants)
├── slurm_submission_scripts/ # SLURM submission scripts for HPC
├── utils/                    # Other utilities (i.e. fid_score_calculator.py)
├── diffusion_training.ipynb  # Interactive training notebook
├── train.py                  # CLI training script
├── requirements.txt          # Dependencies
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delossantos_tan_2025eos,
  title={Edge of Stability in Diffusion Models},
  author={De Los Santos, Arthur and Tan, Viola},
  submission={MIT Statistical Learning Theory 2025 Submission},
  year={2025}
}
```