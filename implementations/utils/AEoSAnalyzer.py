import torch
from pathlib import Path
import csv
# Assuming measure_checkpoint is available as provided in your context
from implementations.utils.measure_checkpoint import EigenvectorCache, compute_eigenvalues, DiagonalPreconditioner

class AEoSAnalyzer:
    def __init__(self, save_dir, filename="aeos_log.csv"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.save_dir / filename
        self.cache = EigenvectorCache(max_eigenvectors=1)

        # Initialize CSV with 'timestep' column
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'timestep', 'batch_sharpness', 'lambda_max', 'lr', 'stability_threshold_38_over_eta', 'average_loss'])

    def get_adam_preconditioner(self, model, optimizer, step):
        """
        Extracts the diagonal preconditioner P from Adam state.
        Adam Update: theta_{t+1} = theta_t - eta * m_t / (sqrt(v_hat_t) + eps)
        This corresponds to a preconditioner P = diag(sqrt(v_hat_t) + eps).
        """
        vals = []
        # Adam hyperparameters
        # We assume all groups have the same betas/eps for simplicity, or take from first group
        beta2 = optimizer.param_groups[0]['betas'][1]
        eps = optimizer.param_groups[0]['eps']
        
        # Bias correction term for v_t
        bias_correction2 = 1 - beta2 ** step

        for p in model.parameters():
            if p.requires_grad:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    v_t = state['exp_avg_sq']
                    # Compute v_hat
                    v_hat = v_t / bias_correction2
                    # P = sqrt(v_hat) + eps
                    p_block = v_hat.flatten().sqrt().add_(eps)
                    vals.append(p_block)
                else:
                    # Fallback if state not initialized (e.g. step 0)
                    # Return Identity (ones)
                    vals.append(torch.ones_like(p).flatten())
        
        if not vals:
            return None
            
        P_vec = torch.cat(vals)
        return DiagonalPreconditioner(P_vec)

    def compute_batch_sharpness(self, loss, model):
        """
        Calculates (g^T * H * g) / ||g||^2
        """
        params = list(model.parameters())
        # 1. Get Gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        grad_norm_sq = torch.dot(grad_flat, grad_flat)

        # 2. HVP of Gradient: H*g
        grad_dot_grad = torch.dot(grad_flat, grad_flat)
        hvp_grads = torch.autograd.grad(grad_dot_grad, params, retain_graph=True)
        hvp_flat = torch.cat([g.reshape(-1) for g in hvp_grads])

        # 3. Rayleigh Quotient
        numerator = torch.dot(grad_flat, hvp_flat)
        sharpness = numerator / (grad_norm_sq + 1e-8)
        return sharpness.item() / 2.0 # divides by 2 because grad g.T g = 2 H g

    def log_step(self, model, optimizer, loss, step, lr, timestep_label="random", average_loss=None):
        """
        Computes Batch Sharpness and Preconditioned Lambda Max.
        """
        # 1. Batch Sharpness (Raw Geometry)
        try:
            bs = self.compute_batch_sharpness(loss, model)
        except Exception as e:
            print(f"Warning: Batch Sharpness failed: {e}")
            bs = 0.0

        # 2. Preconditioned Lambda Max (Adam Stability)
        # We extract P from the optimizer to compute eig(P^{-1} H)
        try:
            # Only construct preconditioner if step > 0 (Adam state exists)
            P = self.get_adam_preconditioner(model, optimizer, step) if step > 0 else None

            lmax = compute_eigenvalues(
                loss, model, k=1, max_iterations=20,
                eigenvector_cache=self.cache,
                P=P  # Pass the Adam preconditioner here
            ).item()
        except Exception as e:
            print(f"Warning: Lambda Max failed: {e}")
            lmax = 0.0

        # 3. Write to CSV
        threshold = 38.0 / lr if lr > 0 else 0
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, timestep_label, bs, lmax, lr, threshold, average_loss])

        return bs, lmax

