"""
Hessian matrix approximation and eigenvalue computation utilities with checkpoint support.
Adapted from measure.py to support loading model checkpoints and computing lambda max.

This module extends measure.py with functionality to:
- Load model checkpoints
- Compute lambda max from a checkpointed model
- Compute eigenvalues of preconditioned Hessian P^{-1} H
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Union, Callable
from .lobpcg import torch_lobpcg, _maybe_orthonormalize


class EigenvectorCache:
    """
    A cache for storing eigenvectors to enable warm starts in power iteration methods.
    Designed to be compatible with future LOBPCG implementations.
    """
    def __init__(self, max_eigenvectors=5):
        self.max_eigenvectors = max_eigenvectors
        self.eigenvectors = []   # List of eigenvectors for multi-eigenvalue computations
        self.eigenvalues = []    # Corresponding eigenvalues
        
    def store_eigenvector(self, eigenvector, eigenvalue=None):
        """Store a single eigenvector (and optionally eigenvalue)"""
        if eigenvalue is not None:
            self.eigenvalues = [eigenvalue]
        self.eigenvectors = [eigenvector]
    
    def store_eigenvectors(self, eigenvectors_list, eigenvalues_list=None):
        """Store multiple eigenvectors (for future LOBPCG compatibility)"""
        self.eigenvectors = [v.detach().clone() for v in eigenvectors_list]
        if eigenvalues_list is not None:
            self.eigenvalues = list(eigenvalues_list)
        
        # Trim to maximum size
        if len(self.eigenvectors) > self.max_eigenvectors:
            self.eigenvectors = self.eigenvectors[:self.max_eigenvectors]
            if self.eigenvalues:
                self.eigenvalues = self.eigenvalues[:self.max_eigenvectors]
    
    def get_warm_start_vectors(self, device=None):
        """Get eigenvectors for warm start, optionally moved to specified device"""
        if not self.eigenvectors:
            return None
        
        if device is not None:
            return [v.to(device) for v in self.eigenvectors]
        return self.eigenvectors
    
    def clear(self):
        """Clear all cached eigenvectors"""
        self.eigenvectors = []
        self.eigenvalues = []
    
    def __len__(self):
        return len(self.eigenvectors)
    
    def __contains__(self, key):
        # For backward compatibility with dict-like access
        return hasattr(self, key) and getattr(self, key) is not None
    
    @property
    def eigenvector(self):
        """Backward compatibility: return first eigenvector if available"""
        if self.eigenvectors:
            return self.eigenvectors[0]
        return None


################################################################################
#                            PRECONDITIONER CLASSES                            #
################################################################################


class Preconditioner:
    """Abstract class for a preconditioner.
    
    A preconditioner P is a linear operator that can be applied to vectors.
    For preconditioned Hessian eigenvalue computation, we need to compute
    eigenvalues of P^{-1} H, where H is the Hessian.
    
    Following centralflows, we compute eigenvalues of the symmetric matrix
    P^{-1/2} H P^{-1/2}, then convert to right eigenvectors of P^{-1} H.
    """
    
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner to a vector.
        
        Args:
            v: The vector to precondition (can be 1D or 2D)
            
        Returns:
            The preconditioned vector P @ v (same shape as v)
        """
        raise NotImplementedError()
    
    def pow(self, p: float) -> 'Preconditioner':
        """Return a new preconditioner which is this preconditioner raised to a power.
        
        Args:
            p: The power (e.g., -1/2 for P^{-1/2})
            
        Returns:
            A new Preconditioner instance
        """
        raise NotImplementedError()


class DiagonalPreconditioner(Preconditioner):
    """A diagonal (elementwise) preconditioner.
    
    For a diagonal preconditioner P, applying it to a vector v means
    elementwise multiplication: (P @ v)[i] = P[i] * v[i]
    
    This is commonly used for optimizers like RMSProp, Adam, etc.
    """
    
    def __init__(self, P: torch.Tensor):
        """Constructor for the diagonal preconditioner.
        
        Args:
            P: The diagonal preconditioner, as a 1D vector of the same length
               as the flattened parameter vector
        """
        self.P = P
    
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the diagonal preconditioner to a vector.
        
        Args:
            v: Vector(s) to precondition. Can be 1D or 2D.
               If 2D, shape is [n_params, k] for k vectors.
               
        Returns:
            Preconditioned vector(s) with same shape as v
        """
        if v.dim() == 1:
            return v * self.P
        elif v.dim() == 2:
            # For 2D input [n_params, k], apply elementwise to each column
            return v * self.P.unsqueeze(1)
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")
    
    def pow(self, power: float) -> 'DiagonalPreconditioner':
        """Return a new diagonal preconditioner raised to a power.
        
        Args:
            power: The power (e.g., -1/2 for P^{-1/2})
            
        Returns:
            A new DiagonalPreconditioner with diagonal P^power
        """
        # Compute P^power
        # For negative powers, we need to be careful with zeros
        if power < 0:
            # P^power = (1/P)^(-power) for power < 0
            # Clamp to avoid division by zero
            P_safe = torch.clamp(self.P, min=1e-12)
            P_pow = (1.0 / P_safe) ** (-power)
        else:
            P_pow = self.P ** power
        
        return DiagonalPreconditioner(P_pow)


class IdentityPreconditioner(Preconditioner):
    """Identity preconditioner (no preconditioning).
    
    This is equivalent to computing eigenvalues of the regular Hessian H.
    """
    
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Return the vector unchanged."""
        return v
    
    def pow(self, power: float) -> 'IdentityPreconditioner':
        """Identity raised to any power is still identity."""
        return self


################################################################################
#                               HELPER FUNCTIONS                               #
################################################################################


def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])


def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return torch.cat([v.flatten() for v in vectors])


################################################################################
#                             EIGENVALUE FUNCTIONS                             #
################################################################################


def create_hessian_vector_product(loss, net, P: Optional[Preconditioner] = None, return_sym_evecs: bool = True):
    """
    Create a Hessian-vector product function for use with LOBPCG.
    
    This function creates a closure that computes the Hessian-vector product
    H @ v where H is the Hessian of the loss function with respect to network parameters.
    
    If a preconditioner P is provided, it computes the matrix-vector product for the
    symmetric preconditioned Hessian P^{-1/2} H P^{-1/2}, which is:
    v -> P^{-1/2} (H (P^{-1/2} v))
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        P (Preconditioner, optional): Preconditioner for computing P^{-1} H eigenvalues.
                                     If None, computes regular Hessian eigenvalues.
        return_sym_evecs (bool): If True and P is provided, returns eigenvectors of
                                P^{-1/2} H P^{-1/2}. If False, the function will be used
                                to compute right eigenvectors of P^{-1} H.
        
    Returns:
        callable: A function that takes a vector v and returns H @ v (or P^{-1/2} H P^{-1/2} @ v)
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)
    
    # Get P^{-1/2} if preconditioner is provided
    if P is not None:
        P_inv_sqrt = P.pow(-1/2)
    else:
        P_inv_sqrt = None
    
    def hessian_vector_product(v):
        """
        Compute Hessian-vector product H @ v (or P^{-1/2} H P^{-1/2} @ v if preconditioned).
        
        Args:
            v (Tensor): Vector(s) to multiply with Hessian. Can be 1D or 2D (for multiple vectors).
            
        Returns:
            Tensor: H @ v (or P^{-1/2} H P^{-1/2} @ v) (same shape as v)
        """
        # Handle both 1D and 2D inputs for compatibility with LOBPCG
        if v.dim() == 1:
            # Single vector case
            # If preconditioned, first apply P^{-1/2} to v
            if P_inv_sqrt is not None:
                v_precond = P_inv_sqrt(v)
            else:
                v_precond = v
            
            # Compute H @ v_precond
            grad_v = torch.dot(grads_vector, v_precond)
            Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
            Hv_flat = flatt(Hv)
            
            # If preconditioned, apply P^{-1/2} again to get P^{-1/2} H P^{-1/2} @ v
            if P_inv_sqrt is not None:
                return P_inv_sqrt(Hv_flat)
            else:
                return Hv_flat
        elif v.dim() == 2:
            # Multiple vectors case (for LOBPCG)
            results = []
            for i in range(v.shape[1]):
                vi = v[:, i]
                
                # If preconditioned, first apply P^{-1/2} to vi
                if P_inv_sqrt is not None:
                    vi_precond = P_inv_sqrt(vi)
                else:
                    vi_precond = vi
                
                # Compute H @ vi_precond
                grad_v = torch.dot(grads_vector, vi_precond)
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                Hv_flat = flatt(Hv)
                
                # If preconditioned, apply P^{-1/2} again
                if P_inv_sqrt is not None:
                    Hv_flat = P_inv_sqrt(Hv_flat)
                
                results.append(Hv_flat)
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")
    
    return hessian_vector_product


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False, P: Optional[Preconditioner] = None,
                                       return_sym_evecs: bool = False):
    """
    Compute multiple eigenvalues of the Hessian (or preconditioned Hessian) using LOBPCG algorithm.
    
    This function computes the top-k eigenvalues of the Hessian matrix using the
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) algorithm.
    
    If a preconditioner P is provided, it computes eigenvalues of P^{-1} H.
    By default, it returns right eigenvectors of P^{-1} H. If return_sym_evecs=True,
    it returns eigenvectors of the symmetric matrix P^{-1/2} H P^{-1/2}.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 5.
        max_iterations (int, optional): Maximum number of LOBPCG iterations. Defaults to 100.
        reltol (float, optional): Relative tolerance for LOBPCG convergence. Defaults to 2% relative tolerance.

        init_vectors (Tensor, optional): Initial vectors for LOBPCG (shape: [n_params, k]). 
                                       If None, uses random or cached vectors.
        eigenvector_cache (EigenvectorCache, optional): Cache for storing/retrieving eigenvectors.
        return_eigenvectors (bool, optional): Whether to return eigenvectors along with eigenvalues.
        P (Preconditioner, optional): Preconditioner for computing P^{-1} H eigenvalues.
                                     If None, computes regular Hessian eigenvalues.
        return_sym_evecs (bool, optional): If True and P is provided, returns eigenvectors of
                                           P^{-1/2} H P^{-1/2}. If False, returns right eigenvectors
                                           of P^{-1} H. Defaults to False.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If return_eigenvectors is False: Returns eigenvalues tensor of shape [k]
            - If return_eigenvectors is True: Returns tuple of (eigenvalues, eigenvectors)
              where eigenvectors has shape [n_params, k]
    
    Note:
        The eigenvalues are returned in descending order (largest first).
        The function automatically handles the case where k is too large relative to the problem size.
    """
    device = next(net.parameters()).device
    n_params = param_length(net)
    
    # Create Hessian-vector product function (for symmetric preconditioned Hessian)
    hessian_matvec = create_hessian_vector_product(loss, net, P=P, return_sym_evecs=True)
    
    # Get P^{-1/2} for converting eigenvectors if needed
    if P is not None:
        P_inv_sqrt = P.pow(-1/2)
    else:
        P_inv_sqrt = None
    
    # Initialize vectors with priority: init_vectors > cached vectors > random
    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k:
            raise ValueError(f"init_vectors must have shape [n_params, {k}], got {X.shape}")
        # If preconditioned and not returning symmetric eigenvectors, convert init vectors
        if P is not None and not return_sym_evecs:
            # Convert right eigenvectors of P^{-1} H to symmetric eigenvectors
            # If u is right eigenvector of P^{-1} H, then P^{1/2} u is eigenvector of P^{-1/2} H P^{-1/2}
            P_sqrt = P.pow(1/2)
            X = torch.stack([P_sqrt(X[:, i]) for i in range(k)], dim=1)
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        # Use cached eigenvectors as initial guess
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            # Take up to k vectors from cache, pad with random if needed
            n_cached = min(len(cached_vectors), k)
            X_list = cached_vectors[:n_cached]
            
            # Pad with random vectors if we don't have enough cached vectors
            if n_cached < k:
                n_random = k - n_cached
                random_vectors = torch.randn(n_params, n_random, device=device)
                X_list.extend([random_vectors[:, i] for i in range(n_random)])
            
            X = torch.stack(X_list, dim=1)
            # Convert if preconditioned and not returning symmetric eigenvectors
            if P is not None and not return_sym_evecs:
                P_sqrt = P.pow(1/2)
                X = torch.stack([P_sqrt(X[:, i]) for i in range(k)], dim=1)
        else:
            X = torch.randn(n_params, k, device=device)
    else:
        # Use random initialization
        X = torch.randn(n_params, k, device=device)
    
    # Ensure X is on the correct device and has the right shape
    X = X.to(device)
    if X.shape != (n_params, k):
        X = X.reshape(n_params, k)
    
    # Run LOBPCG (computes eigenvalues of P^{-1/2} H P^{-1/2})
    tol = reltol / (20 * n_params)  # Adjust tolerance based on problem size

    eigenvalues, eigenvectors, iterations = torch_lobpcg(
        hessian_matvec, X, max_iter=max_iterations, tol=tol
    )
    
    # Convert eigenvectors if needed
    # eigenvectors are currently eigenvectors of P^{-1/2} H P^{-1/2}
    # If return_sym_evecs=False, convert to right eigenvectors of P^{-1} H
    if P is not None and not return_sym_evecs:
        # Convert: right eigenvector of P^{-1} H = P^{-1/2} * (eigenvector of P^{-1/2} H P^{-1/2})
        eigenvectors = torch.stack([P_inv_sqrt(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])], dim=1)
    
    # Store eigenvectors in cache for future use
    if eigenvector_cache is not None:
        eigenvector_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigenvector_list, eigenvalues.tolist())
    
    # Return results
    if return_eigenvectors:
        return eigenvalues, eigenvectors
    else:
        return eigenvalues


def compute_lambdamax_power_iteration(loss, net, max_iterations, reltol, init_vector,
                                       eigenvector_cache, return_eigenvector):
    """Power iteration implementation of the maximum eigenvalue of the Hessian."""
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)

    size = param_length(net)
    
    # Initialize vector with priority: init_vector > cached eigenvector > random
    if init_vector is not None:
        v = init_vector
    elif eigenvector_cache is not None:
        # Support both EigenvectorCache objects and dict-style caches
        if isinstance(eigenvector_cache, EigenvectorCache):
            if len(eigenvector_cache) > 0:
                cached_v = eigenvector_cache.eigenvector
                if cached_v.device != device:
                    cached_v = cached_v.to(device)
                v = cached_v.detach()
            else:
                v = torch.randn(size, device=device)
        elif isinstance(eigenvector_cache, dict) and 'eigenvector' in eigenvector_cache:
            # Backward compatibility with dict-style cache
            cached_v = eigenvector_cache['eigenvector']
            if cached_v.device != device:
                cached_v = cached_v.to(device)
            v = cached_v.detach()
        else:
            v = torch.randn(size, device=device)
    else:
        # Use random vector as initial vector instead of gradient
        v = torch.randn(size, device=device)
    
    with torch.no_grad():
        v = v / torch.linalg.norm(v)

    # Power iteration method to find the maximum eigenvalue
    v = v.detach()
    eigenval = 0.0  # Initialize eigenval to avoid undefined variable error
    for i in range(max_iterations):
        # grad_vector = \nabla L
        grad_v = torch.dot(grads_vector, v) # \nabla L . v
        Hv = flatt(torch.autograd.grad(grad_v, params, retain_graph=True)).detach() # \nabla (\nabla L . v) = H(L) * v

        v = v.detach()
        with torch.no_grad():
            rayleigh_quotient = torch.dot(Hv, v) / torch.dot(v, v)
            eigenval = rayleigh_quotient  # Update eigenval every iteration
            if torch.abs(rayleigh_quotient) < 1e-12:
                break

            residual = Hv - rayleigh_quotient * v
            resid_norm = torch.linalg.norm(residual)
            if resid_norm / torch.abs(rayleigh_quotient) < reltol:
                break
            
            v = Hv / torch.linalg.norm(Hv) # Normalize for next iteration

    # Store the final eigenvector in cache for future warm starts
    if eigenvector_cache is not None:
        if isinstance(eigenvector_cache, EigenvectorCache):
            eigenvector_cache.store_eigenvector(v, eigenval)
        else:
            raise ValueError("eigenvector_cache must be an instance of EigenvectorCache")

    # Prepare return values
    results = [eigenval]
    
    if return_eigenvector:
        results.append(v.detach())
    
    # Return single value or tuple based on what was requested
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def compute_eigenvalues(loss, 
                        net, 
                        k=1, 
                        max_iterations=100, 
                        reltol=1e-2,
                        init_vectors=None,
                        batched=None,
                        eigenvector_cache=None,
                        return_eigenvectors: bool = False,
                        use_power_iteration: bool = False,
                        P: Optional[Preconditioner] = None,
                        return_sym_evecs: bool = False):
    """
    Computes the top-k eigenvalues of the Hessian (or preconditioned Hessian) of the loss function at the current point.
    
    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.
    
    If a preconditioner P is provided, it computes eigenvalues of P^{-1} H.
    By default, it returns right eigenvectors of P^{-1} H. If return_sym_evecs=True,
    it returns eigenvectors of the symmetric matrix P^{-1/2} H P^{-1/2}.

    Args:
        loss (Tensor): The loss value at the current point
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
        reltol (float, optional): relative tolerance threshold for eigenvalue computation. Defaults to 1e-2.
        init_vectors (Tensor, optional): Initial vectors. For k=1, can be 1D vector. For k>1, should be [n_params, k]. 
                                        If None, uses cached or random vectors. Defaults to None.
        batched (Any, optional): Unused parameter. Defaults to None.
        eigenvector_cache (EigenvectorCache, optional): Cache to store/retrieve eigenvectors for warm starts. Defaults to None.
        return_eigenvectors (bool, optional): Whether to return the final eigenvectors. Defaults to False.
        use_power_iteration (bool, optional): If True, force use of power iteration (only works for k=1). Defaults to False.
        P (Preconditioner, optional): Preconditioner for computing P^{-1} H eigenvalues.
                                     If None, computes regular Hessian eigenvalues.
        return_sym_evecs (bool, optional): If True and P is provided, returns eigenvectors of
                                           P^{-1/2} H P^{-1/2}. If False, returns right eigenvectors
                                           of P^{-1} H. Defaults to False.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If k=1 and return_eigenvectors=False: Returns single eigenvalue (scalar Tensor)
            - If k=1 and return_eigenvectors=True: Returns (eigenvalue, eigenvector)
            - If k>1 and return_eigenvectors=False: Returns eigenvalues tensor of shape [k]
            - If k>1 and return_eigenvectors=True: Returns (eigenvalues, eigenvectors) where 
              eigenvalues has shape [k] and eigenvectors has shape [n_params, k]

    Note:
        By default, uses LOBPCG for eigenvalue computation for better performance.
        Falls back to power iteration if use_power_iteration=True (only supported for k=1).
        
        If eigenvector_cache is provided, the function will try to reuse previous eigenvectors
        for warm starts and store the final eigenvector(s) for future use.
        
        Preconditioning is only supported with LOBPCG (not power iteration).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")
    
    if use_power_iteration and P is not None:
        raise ValueError("Preconditioning is not supported with power iteration. Use LOBPCG (default) instead.")
    
    device = next(net.parameters()).device

    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        # Use the existing power iteration implementation
        # Handle init_vectors: if it's a 2D tensor, take first column; if 1D, use as is; if None, pass None
        init_vector = None
        if init_vectors is not None:
            if init_vectors.dim() == 2:
                init_vector = init_vectors[:, 0]  # Take first column for k=1
            elif init_vectors.dim() == 1:
                init_vector = init_vectors
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vector,
            eigenvector_cache, return_eigenvectors
        )
    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors, 
            eigenvector_cache, return_eigenvectors=True, P=P, return_sym_evecs=return_sym_evecs
        )
        
        if k == 1:
            # For backward compatibility with single eigenvalue case
            eigenvalue = eigenvalues[0]
            if return_eigenvectors:
                return eigenvalue, eigenvectors[:, 0]
            else:
                return eigenvalue
        else:
            # Multiple eigenvalues case
            if return_eigenvectors:
                return eigenvalues, eigenvectors
            else:
                return eigenvalues


################################################################################
#                         CHECKPOINT-BASED FUNCTIONS                           #
################################################################################


def load_checkpoint(checkpoint_path: Union[str, Path], 
                    model: nn.Module,
                    device: Optional[torch.device] = None,
                    use_ema: bool = False) -> dict:
    """
    Load a model checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt or .pth)
        model: The model instance to load weights into
        device: Device to load the checkpoint on. If None, uses model's device.
        use_ema: If True, load EMA weights instead of regular model weights.
        
    Returns:
        dict: Checkpoint dictionary containing model_state, ema_state, epoch, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle torch.compile wrapped models (remove _orig_mod prefix)
    def remove_orig_mod_prefix(state_dict):
        """Remove _orig_mod. prefix from keys if present (from torch.compile)"""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    # Load model weights with strict=False to handle missing keys (e.g., attention layers)
    missing_keys = []
    unexpected_keys = []
    
    if use_ema and 'ema_state' in checkpoint:
        ema_state = remove_orig_mod_prefix(checkpoint['ema_state'])
        missing_keys, unexpected_keys = model.load_state_dict(ema_state, strict=False)
        print(f"Loaded EMA weights from checkpoint: {checkpoint_path}")
        if missing_keys:
            print(f"  Warning: Missing keys (not loaded): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    elif 'model_state' in checkpoint:
        model_state = remove_orig_mod_prefix(checkpoint['model_state'])
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        print(f"Loaded model weights from checkpoint: {checkpoint_path}")
        if missing_keys:
            print(f"  Warning: Missing keys (not loaded): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    else:
        # Try loading directly (some checkpoints might not have 'model_state' key)
        try:
            direct_state = remove_orig_mod_prefix(checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(direct_state, strict=False)
            print(f"Loaded model weights directly from checkpoint: {checkpoint_path}")
            if missing_keys:
                print(f"  Warning: Missing keys (not loaded): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Warning: Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        except Exception as e:
            raise ValueError(f"Could not load model weights from checkpoint. Available keys: {checkpoint.keys()}. Error: {e}")
    
    # ALWAYS replace attention layers with identity to avoid Hessian computation issues
    # PyTorch's scaled_dot_product_attention doesn't support second-order derivatives
    # This is safe because Identity preserves input shape and supports second-order derivatives
    # We do this unconditionally because we're computing Hessians, not doing inference
    attention_layer_names = ['down2_attn', 'mid_attn', 'up1_attn']
    for attn_name in attention_layer_names:
        if hasattr(model, attn_name):
            # Always replace attention layer with identity for Hessian computation
            # This avoids the "derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented" error
            setattr(model, attn_name, nn.Identity())
            print(f"  Replaced {attn_name} with Identity (avoids Hessian computation issues with scaled_dot_product_attention)")
    
    model.to(device)
    model.eval()  # Set to eval mode by default
    
    return checkpoint


def compute_lambda_max_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    loss_fn: Callable,
    data_sample: torch.Tensor,
    device: Optional[torch.device] = None,
    use_ema: bool = False,
    k: int = 1,
    max_iterations: int = 100,
    reltol: float = 1e-2,
    use_power_iteration: bool = False,
    return_eigenvectors: bool = False,
    eigenvector_cache: Optional[EigenvectorCache] = None,
    P: Optional[Preconditioner] = None,
    return_sym_evecs: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute lambda max (largest eigenvalue) from a model checkpoint.
    
    This function loads a checkpoint, computes the loss on the provided data sample,
    and then computes the largest eigenvalue(s) of the Hessian (or preconditioned Hessian).
    
    If a preconditioner P is provided, it computes eigenvalues of P^{-1} H.
    By default, it returns right eigenvectors of P^{-1} H. If return_sym_evecs=True,
    it returns eigenvectors of the symmetric matrix P^{-1/2} H P^{-1/2}.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model instance (architecture). Will be loaded with checkpoint weights.
        loss_fn: Loss function that takes (model, *args) and returns a loss tensor.
                 For diffusion models with min SNR reweighting, this would be something like:
                 lambda model, schedule, x, t: min_snr_loss(model, schedule, x, t, snr_gamma=5.0)
        data_sample: Input data tensor for computing the loss. Shape should match model input.
        device: Device to run computation on. If None, uses model's device or cuda if available.
        use_ema: If True, load EMA weights from checkpoint instead of regular weights.
        k: Number of eigenvalues to compute. Defaults to 1 (lambda max).
        max_iterations: Maximum number of iterations for eigenvalue computation.
        reltol: Relative tolerance for convergence.
        use_power_iteration: If True, use power iteration (only works for k=1, not supported with preconditioning).
        return_eigenvectors: If True, return eigenvectors along with eigenvalues.
        eigenvector_cache: Optional cache for warm starts.
        P (Preconditioner, optional): Preconditioner for computing P^{-1} H eigenvalues.
                                     If None, computes regular Hessian eigenvalues.
        return_sym_evecs (bool, optional): If True and P is provided, returns eigenvectors of
                                           P^{-1/2} H P^{-1/2}. If False, returns right eigenvectors
                                           of P^{-1} H. Defaults to False.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If k=1 and return_eigenvectors=False: Returns lambda max (scalar Tensor)
            - If k=1 and return_eigenvectors=True: Returns (lambda_max, eigenvector)
            - If k>1 and return_eigenvectors=False: Returns eigenvalues tensor of shape [k]
            - If k>1 and return_eigenvectors=True: Returns (eigenvalues, eigenvectors)
    
    Example:
        >>> from implementations.base_implementation import UNet
        >>> from implementations.min_snr_reweighting import MinSNRDiffusionSchedule, min_snr_loss
        >>> from implementations.utils.measure_checkpoint_min_snr import (
        ...     create_preconditioner_from_checkpoint, create_constant_preconditioner
        ... )
        >>> import torch
        >>> 
        >>> # Create model and schedule
        >>> model = UNet()
        >>> schedule = MinSNRDiffusionSchedule(timesteps=50)
        >>> 
        >>> # Create data sample
        >>> x_sample = torch.randn(4, 3, 32, 32)  # batch_size=4, CIFAR-10
        >>> t_sample = torch.randint(0, 50, (4,))
        >>> 
        >>> # Option 1: Extract preconditioner from checkpoint (if optimizer state is saved)
        >>> P = create_preconditioner_from_checkpoint(
        ...     checkpoint_path='./runs/checkpoint_epoch_10.pt',
        ...     model=model,
        ...     optimizer_type='auto'  # or 'adam', 'rmsprop', 'sgd'
        ... )
        >>> 
        >>> # Option 2: Create constant preconditioner (for SGD or if optimizer state not available)
        >>> if P is None:
        ...     P = create_constant_preconditioner(model, lr=2e-4)
        >>> 
        >>> # Compute lambda max of preconditioned Hessian
        >>> lambda_max = compute_lambda_max_from_checkpoint(
        ...     checkpoint_path='./runs/checkpoint_epoch_10.pt',
        ...     model=model,
        ...     loss_fn=lambda m, s, x, t: min_snr_loss(m, s, x, t, snr_gamma=5.0),
        ...     data_sample=(schedule, x_sample, t_sample),
        ...     device='cuda',
        ...     P=P
        ... )
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device, use_ema=use_ema)
    
    # Ensure model requires gradients for Hessian computation
    for param in model.parameters():
        param.requires_grad = True
    
    # Prepare data sample
    # Handle both tuple and single tensor inputs
    if isinstance(data_sample, tuple):
        # Unpack tuple for loss function
        with torch.enable_grad():
            loss = loss_fn(model, *data_sample)
    else:
        # Single tensor input
        data_sample = data_sample.to(device)
        with torch.enable_grad():
            loss = loss_fn(model, data_sample)
    
    # Compute eigenvalues
    result = compute_eigenvalues(
        loss=loss,
        net=model,
        k=k,
        max_iterations=max_iterations,
        reltol=reltol,
        eigenvector_cache=eigenvector_cache,
        return_eigenvectors=return_eigenvectors,
        use_power_iteration=use_power_iteration,
        P=P,
        return_sym_evecs=return_sym_evecs
    )
    
    return result


def compute_lambda_max_from_checkpoint_simple(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    schedule,
    x_sample: torch.Tensor,
    t_sample: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    use_ema: bool = False,
    k: int = 1,
    max_iterations: int = 100,
    reltol: float = 1e-2,
    use_power_iteration: bool = False,
    return_eigenvectors: bool = False,
    P: Optional[Preconditioner] = None,
    return_sym_evecs: bool = False,
    use_preconditioner: bool = False,
    optimizer_type: str = 'auto',
    learning_rate: Optional[float] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Simplified version for diffusion models that automatically handles the loss function.
    
    This is a convenience wrapper around compute_lambda_max_from_checkpoint specifically
    for diffusion models using the min_snr_loss function (Min-SNR reweighting).
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model instance (UNet architecture)
        schedule: MinSNRDiffusionSchedule instance
        x_sample: Input data tensor (batch_size, channels, height, width)
        t_sample: Timestep tensor (batch_size,). If None, random timesteps are generated.
        device: Device to run computation on
        use_ema: If True, load EMA weights from checkpoint
        k: Number of eigenvalues to compute (default: 1 for lambda max)
        max_iterations: Maximum iterations for eigenvalue computation
        reltol: Relative tolerance
        use_power_iteration: Use power iteration instead of LOBPCG (not supported with preconditioning)
        return_eigenvectors: Return eigenvectors along with eigenvalues
        P (Preconditioner, optional): Preconditioner for computing P^{-1} H eigenvalues.
                                     If None, computes regular Hessian eigenvalues.
        return_sym_evecs (bool, optional): If True and P is provided, returns eigenvectors of
                                           P^{-1/2} H P^{-1/2}. If False, returns right eigenvectors
                                           of P^{-1} H. Defaults to False.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: Lambda max or (eigenvalues, eigenvectors)
    
    Example:
        >>> from implementations.base_implementation import UNet
        >>> from implementations.min_snr_reweighting import MinSNRDiffusionSchedule
        >>> from implementations.utils.measure_checkpoint_min_snr import (
        ...     create_preconditioner_from_checkpoint, create_constant_preconditioner
        ... )
        >>> import torch
        >>> 
        >>> model = UNet()
        >>> schedule = MinSNRDiffusionSchedule(timesteps=50)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> 
        >>> # Extract preconditioner from checkpoint (or create constant one)
        >>> P = create_preconditioner_from_checkpoint(
        ...     './runs/checkpoint_epoch_10.pt', model, optimizer_type='auto'
        ... )
        >>> if P is None:
        ...     P = create_constant_preconditioner(model, lr=2e-4)
        >>> 
        >>> lambda_max = compute_lambda_max_from_checkpoint_simple(
        ...     './runs/checkpoint_epoch_10.pt',
        ...     model, schedule, x, P=P
        ... )
    """
    # Import min_snr_loss here to avoid circular imports
    try:
        from ..min_snr_reweighting import min_snr_loss
    except ImportError:
        # Fallback: define min_snr_loss locally if import fails
        def min_snr_loss(model, schedule, x_start, t, snr_gamma=5.0):
            noise = torch.randn_like(x_start)
            x_noisy = schedule.q_sample(x_start=x_start, t=t, noise=noise)
            predicted_noise = model(x_noisy, t)
            
            # Calculate MSE per sample
            # Shape: [B, C, H, W] -> [B]
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
            loss = loss.mean(dim=[1, 2, 3])
            
            # Get SNR for the current batch timesteps
            snr = schedule.snr[t]
            
            # Calculate weights: w = min(1, gamma / SNR)
            mse_loss_weights = torch.clamp(snr_gamma / snr, max=1.0)
            
            # Apply weights and average over batch
            return (loss * mse_loss_weights).mean()
    
    # Generate timesteps if not provided
    if t_sample is None:
        batch_size = x_sample.shape[0]
        t_sample = torch.randint(0, schedule.timesteps, (batch_size,))
    elif isinstance(t_sample, (int, float)) or (isinstance(t_sample, torch.Tensor) and t_sample.dim() == 0):
        # Convert scalar to tensor with shape (batch_size,)
        batch_size = x_sample.shape[0]
        if isinstance(t_sample, torch.Tensor):
            t_sample = t_sample.item()
        t_sample = torch.full((batch_size,), int(t_sample), dtype=torch.long)
    
    # Move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_sample = x_sample.to(device)
    t_sample = t_sample.to(device)
    schedule.device = device
    
    # If use_preconditioner is True but P is None, create it after loading checkpoint
    # This ensures the preconditioner size matches the loaded model
    if use_preconditioner and P is None:
        # Load checkpoint first to get the actual model size
        temp_checkpoint = load_checkpoint(checkpoint_path, model, device=device, use_ema=use_ema)
        
        # Now create preconditioner with the loaded model
        P = create_preconditioner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer_type=optimizer_type,
            lr=learning_rate,
            device=device
        )
        
        # Fallback to constant preconditioner if optimizer state not available
        if P is None:
            if learning_rate is None:
                print(f"  ⚠️  Warning: No optimizer state found and no learning rate provided.")
                print(f"     Skipping preconditioner. Computing regular lambda max.")
            else:
                print(f"  No optimizer state found. Creating constant preconditioner with lr={learning_rate}")
                P = create_constant_preconditioner(model, lr=learning_rate, device=device)
    
    # Use the general function
    # Note: min_snr_loss requires snr_gamma parameter, default is 5.0
    return compute_lambda_max_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        loss_fn=lambda m, s, x, t: min_snr_loss(m, s, x, t, snr_gamma=5.0),
        data_sample=(schedule, x_sample, t_sample),
        device=device,
        use_ema=use_ema,
        k=k,
        max_iterations=max_iterations,
        reltol=reltol,
        use_power_iteration=use_power_iteration,
        return_eigenvectors=return_eigenvectors,
        P=P,
        return_sym_evecs=return_sym_evecs
    )


################################################################################
#                    PRECONDITIONER EXTRACTION UTILITIES                      #
################################################################################


def extract_preconditioner_from_optimizer_state(
    optimizer_state_dict: dict,
    model: nn.Module,
    optimizer_type: str = 'auto',
    lr: Optional[float] = None,
    eps: float = 1e-8,
    beta2: Optional[float] = None
) -> DiagonalPreconditioner:
    """
    Extract a diagonal preconditioner from PyTorch optimizer state.
    
    This function extracts the preconditioner P from common PyTorch optimizers
    (Adam, RMSProp, SGD) based on their state_dict. The preconditioner represents
    the effective step size matrix used by the optimizer.
    
    For SGD: P = (1/lr) * I (constant diagonal)
    For Adam: P[i] = sqrt(exp_avg_sq[i] + eps) / lr
    For RMSProp: P[i] = sqrt(square_avg[i] + eps) / lr
    
    Args:
        optimizer_state_dict: The optimizer's state_dict (from optimizer.state_dict()['state'])
        model: The model to extract parameter shapes from
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'sgd', or 'auto' to infer)
        lr: Learning rate (required for SGD, optional for others if in state)
        eps: Epsilon value for numerical stability (default: 1e-8)
        beta2: Beta2 parameter for Adam/RMSProp (unused, kept for compatibility)
        
    Returns:
        DiagonalPreconditioner: The extracted preconditioner
        
    Note:
        The preconditioner P is such that the update is: w = w - P^{-1} @ grad
        For SGD, this is equivalent to: w = w - lr * grad
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    
    # Auto-detect optimizer type if needed
    if optimizer_type == 'auto':
        if len(optimizer_state_dict) == 0:
            raise ValueError("Optimizer state_dict is empty. Cannot infer optimizer type.")
        
        # Get first parameter's state
        first_param_state = list(optimizer_state_dict.values())[0]
        
        if 'exp_avg_sq' in first_param_state:
            optimizer_type = 'adam'
        elif 'square_avg' in first_param_state:
            optimizer_type = 'rmsprop'
        elif 'momentum_buffer' in first_param_state:
            optimizer_type = 'sgd'
        else:
            # Default to SGD if we can't determine
            optimizer_type = 'sgd'
    
    optimizer_type = optimizer_type.lower()
    
    # Extract preconditioner diagonal
    # Note: optimizer state_dict keys can be:
    # 1. Parameter IDs (integers) when loaded from checkpoint directly
    # 2. Parameter objects (torch.nn.Parameter) when from a live optimizer
    
    # Validate that we have the right structure - keys should be integers or parameter objects
    # If we see 'state' or 'param_groups' as keys, we got the wrong structure
    sample_keys = list(optimizer_state_dict.keys())[:5]
    if any(key in ['state', 'param_groups'] for key in sample_keys):
        raise ValueError(
            f"Received full optimizer state_dict instead of just the 'state' part. "
            f"Keys found: {sample_keys}. "
            f"This is a bug in create_preconditioner_from_checkpoint - it should extract optimizer_state_dict['state'] "
            f"before calling this function."
        )
    
    # Match states to parameters
    # If keys are parameter objects, match directly
    # If keys are integers, match by order
    state_values = []
    
    # Check if keys are parameter objects or integers
    first_key = list(optimizer_state_dict.keys())[0] if len(optimizer_state_dict) > 0 else None
    
    if first_key is not None and isinstance(first_key, torch.nn.Parameter):
        # Keys are parameter objects - match directly
        param_list = list(params)
        for param in param_list:
            if param in optimizer_state_dict:
                state_values.append(optimizer_state_dict[param])
            else:
                # Parameter not in optimizer state (e.g., newly added layer)
                # Create empty state dict for fallback
                state_values.append({})
    else:
        # Keys are integers (parameter IDs) - match by order
        # Convert keys to integers and sort them
        state_items = []
        for key, value in optimizer_state_dict.items():
            if isinstance(key, (int, str)):
                try:
                    param_id = int(key)
                    state_items.append((param_id, value))
                except (ValueError, TypeError):
                    continue
            else:
                state_items.append((key, value))
        
        # Sort by parameter ID to match parameter order
        try:
            state_items.sort(key=lambda x: x[0] if isinstance(x[0], int) else 0)
        except TypeError:
            pass
        
        state_values = [value for _, value in state_items]
    
    if len(state_values) != len(params):
        # Provide more diagnostic information
        state_keys = list(optimizer_state_dict.keys())[:10]  # First 10 keys for debugging
        raise ValueError(
            f"Number of parameters in model ({len(params)}) doesn't match "
            f"number of states in optimizer ({len(state_values)}). "
            f"Optimizer state_dict has {len(optimizer_state_dict)} entries. "
            f"First few state keys: {state_keys}. "
            "This might happen if:\n"
            "1. The checkpoint was saved with a different model architecture\n"
            "2. The optimizer state is incomplete or corrupted\n"
            "3. The checkpoint doesn't contain optimizer state (try setting use_preconditioner=False or provide learning_rate for constant preconditioner)"
        )
    
    P_diag_list = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    if optimizer_type == 'adam':
        for i, (param, state) in enumerate(zip(params, state_values)):
            exp_avg_sq = state.get('exp_avg_sq', None)
            if exp_avg_sq is None:
                # Fallback: use constant preconditioner based on learning rate
                current_lr = lr if lr is not None else state.get('lr', 1e-3)
                P_diag = torch.ones(param.numel(), device=device, dtype=dtype) / current_lr
            else:
                # Validate shape matches parameter
                if exp_avg_sq.shape != param.shape:
                    # Shape mismatch - use fallback
                    current_lr = lr if lr is not None else state.get('lr', 1e-3)
                    P_diag = torch.ones(param.numel(), device=device, dtype=dtype) / current_lr
                else:
                    # P = sqrt(exp_avg_sq + eps) / lr
                    # The preconditioner P is such that the update is: w = w - P^{-1} @ grad
                    current_lr = lr if lr is not None else state.get('lr', 1e-3)
                    # Ensure exp_avg_sq is on the right device
                    if exp_avg_sq.device != device:
                        exp_avg_sq = exp_avg_sq.to(device)
                    if exp_avg_sq.dtype != dtype:
                        exp_avg_sq = exp_avg_sq.to(dtype)
                    P_diag = torch.sqrt(exp_avg_sq + eps) / current_lr
            P_diag_list.append(P_diag.flatten())
    
    elif optimizer_type == 'rmsprop':
        for i, (param, state) in enumerate(zip(params, state_values)):
            square_avg = state.get('square_avg', None)
            if square_avg is None:
                current_lr = lr if lr is not None else state.get('lr', 1e-3)
                P_diag = torch.ones(param.numel(), device=device, dtype=dtype) / current_lr
            else:
                # Validate shape matches parameter
                if square_avg.shape != param.shape:
                    # Shape mismatch - use fallback
                    current_lr = lr if lr is not None else state.get('lr', 1e-3)
                    P_diag = torch.ones(param.numel(), device=device, dtype=dtype) / current_lr
                else:
                    current_lr = lr if lr is not None else state.get('lr', 1e-3)
                    if square_avg.device != device:
                        square_avg = square_avg.to(device)
                    if square_avg.dtype != dtype:
                        square_avg = square_avg.to(dtype)
                    P_diag = torch.sqrt(square_avg + eps) / current_lr
            P_diag_list.append(P_diag.flatten())
    
    elif optimizer_type == 'sgd':
        if lr is None:
            raise ValueError("Learning rate (lr) is required for SGD preconditioner")
        
        # SGD has constant preconditioner: P = 1 / lr
        P_diag = torch.ones(n_params, device=device, dtype=dtype) / lr
        return DiagonalPreconditioner(P_diag)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'adam', 'rmsprop', 'sgd'")
    
    # Concatenate all parameter preconditioners
    P_diag = torch.cat(P_diag_list)
    
    return DiagonalPreconditioner(P_diag)


def create_preconditioner_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer_type: str = 'auto',
    lr: Optional[float] = None,
    eps: float = 1e-8,
    device: Optional[torch.device] = None
) -> Optional[DiagonalPreconditioner]:
    """
    Create a preconditioner from a checkpoint file if it contains optimizer state.
    
    This function creates an optimizer, loads its state from the checkpoint, and then
    extracts the preconditioner. This ensures proper parameter matching.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model instance
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'sgd', or 'auto' to infer)
        lr: Learning rate (required for SGD, optional for others - will try to extract from param_groups)
        eps: Epsilon value for numerical stability
        device: Device to load checkpoint on
        
    Returns:
        DiagonalPreconditioner if optimizer state found, None otherwise
        
    Note:
        The checkpoint must contain optimizer state. If your checkpoints don't include
        optimizer state, you'll need to either:
        1. Modify your training code to save optimizer state:
            ckpt = {
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                ...
            }
        2. Or create a constant preconditioner using create_constant_preconditioner()
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get optimizer state_dict from checkpoint
    optimizer_state_dict = None
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state_dict = checkpoint['optimizer_state_dict']
    elif 'optimizer_state' in checkpoint:
        optimizer_state_dict = checkpoint['optimizer_state']
    
    if optimizer_state_dict is None:
        # No optimizer state in checkpoint
        return None
    
    # Try to extract learning rate from param_groups if not provided
    if lr is None:
        if isinstance(optimizer_state_dict, dict) and 'param_groups' in optimizer_state_dict:
            param_groups = optimizer_state_dict['param_groups']
            if len(param_groups) > 0:
                lr = param_groups[0].get('lr', None)
    
    # Auto-detect optimizer type if needed
    if optimizer_type == 'auto':
        if isinstance(optimizer_state_dict, dict) and 'state' in optimizer_state_dict:
            state_dict = optimizer_state_dict['state']
            if len(state_dict) > 0:
                first_state = list(state_dict.values())[0]
                if 'exp_avg_sq' in first_state:
                    optimizer_type = 'adam'
                elif 'square_avg' in first_state:
                    optimizer_type = 'rmsprop'
                elif 'momentum_buffer' in first_state:
                    optimizer_type = 'sgd'
                else:
                    optimizer_type = 'adam'  # Default to Adam
        else:
            # Can't infer, default to Adam
            optimizer_type = 'adam'
    
    optimizer_type = optimizer_type.lower()
    
    # Extract the state part from optimizer_state_dict
    if isinstance(optimizer_state_dict, dict) and 'state' in optimizer_state_dict:
        # Full optimizer state_dict with 'state' and 'param_groups'
        checkpoint_state = optimizer_state_dict['state']
        param_groups = optimizer_state_dict.get('param_groups', [])
    else:
        # Just the state part
        checkpoint_state = optimizer_state_dict
        param_groups = []
    
    # Get model parameters and their names
    model_params = list(model.parameters())
    model_param_names = [name for name, _ in model.named_parameters()]
    n_model_params = len(model_params)
    
    # Get number of states in checkpoint
    n_checkpoint_states = len(checkpoint_state)
    
    # Load the checkpoint model state_dict to see which parameters were successfully loaded
    # This tells us which optimizer states correspond to parameters that still exist
    checkpoint_model_state = None
    if 'model_state' in checkpoint:
        checkpoint_model_state = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        checkpoint_model_state = checkpoint['model_state_dict']
    
    # Create optimizer with the model
    # Use a dummy learning rate if not provided (will be overridden by state_dict)
    if lr is None:
        lr = 1e-3  # Dummy value, will be overridden
    
    try:
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Strategy: Match optimizer states to model parameters by checking which parameters
        # from the checkpoint model still exist in the current model
        # We'll match states to parameters that have matching shapes and exist in both models
        
        # Get checkpoint parameter names (if available)
        checkpoint_param_names = None
        if checkpoint_model_state is not None:
            checkpoint_param_names = list(checkpoint_model_state.keys())
        
        # Create filtered state dict mapping model parameters to checkpoint states
        filtered_state = {}
        checkpoint_keys_sorted = sorted(checkpoint_state.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) else 0)
        
        # Match states to parameters: try to match by order first, but validate shapes
        # If we have parameter names, we can be smarter about matching
        matched_indices = set()
        
        for i, (param_obj, param_name) in enumerate(zip(model_params, model_param_names)):
            # Try to find a matching checkpoint state
            best_match_idx = None
            best_match_state = None
            
            # First, try matching by position (if indices haven't been used)
            if i < len(checkpoint_keys_sorted) and i not in matched_indices:
                checkpoint_key = checkpoint_keys_sorted[i]
                checkpoint_param_state = checkpoint_state[checkpoint_key]
                
                # Check if shapes match
                shape_match = False
                if 'exp_avg_sq' in checkpoint_param_state:
                    shape_match = (checkpoint_param_state['exp_avg_sq'].shape == param_obj.shape)
                elif 'square_avg' in checkpoint_param_state:
                    shape_match = (checkpoint_param_state['square_avg'].shape == param_obj.shape)
                elif 'momentum_buffer' in checkpoint_param_state:
                    shape_match = (checkpoint_param_state['momentum_buffer'].shape == param_obj.shape)
                
                if shape_match:
                    best_match_idx = i
                    best_match_state = checkpoint_param_state
            
            # If no match by position, try to find any unused state with matching shape
            if best_match_state is None:
                for j, checkpoint_key in enumerate(checkpoint_keys_sorted):
                    if j in matched_indices:
                        continue
                    checkpoint_param_state = checkpoint_state[checkpoint_key]
                    
                    # Check if shapes match
                    shape_match = False
                    if 'exp_avg_sq' in checkpoint_param_state:
                        shape_match = (checkpoint_param_state['exp_avg_sq'].shape == param_obj.shape)
                    elif 'square_avg' in checkpoint_param_state:
                        shape_match = (checkpoint_param_state['square_avg'].shape == param_obj.shape)
                    elif 'momentum_buffer' in checkpoint_param_state:
                        shape_match = (checkpoint_param_state['momentum_buffer'].shape == param_obj.shape)
                    
                    if shape_match:
                        best_match_idx = j
                        best_match_state = checkpoint_param_state
                        break
            
            # If we found a match, use it
            if best_match_state is not None:
                filtered_state[param_obj] = best_match_state
                if best_match_idx is not None:
                    matched_indices.add(best_match_idx)
        
        # Create the filtered optimizer state_dict
        optimizer_state_dict_filtered = {
            'state': filtered_state,
            'param_groups': optimizer.state_dict()['param_groups']  # Keep param_groups from new optimizer
        }
        
        # Only proceed if we matched at least some states
        if len(filtered_state) == 0:
            import warnings
            warnings.warn(
                f"Could not match any optimizer states to model parameters. "
                f"Model has {n_model_params} parameters, checkpoint has {n_checkpoint_states} states. "
                f"Falling back to constant preconditioner (if learning_rate is provided)."
            )
            return None
        
        # Load the filtered optimizer state_dict
        try:
            optimizer.load_state_dict(optimizer_state_dict_filtered)
        except Exception as e:
            # If loading still fails, return None to trigger fallback
            import warnings
            warnings.warn(
                f"Could not load filtered optimizer state_dict: {e}\n"
                f"Matched {len(filtered_state)} out of {n_model_params} parameters. "
                f"Falling back to constant preconditioner (if learning_rate is provided)."
            )
            return None
        
        # Now extract preconditioner from the loaded optimizer
        # The optimizer.state_dict()['state'] contains the parameter states
        optimizer_state = optimizer.state_dict()['state']
        
        return extract_preconditioner_from_optimizer_state(
            optimizer_state, model, optimizer_type, lr, eps
        )
    except Exception as e:
        # If anything fails, return None to trigger fallback
        import warnings
        warnings.warn(
            f"Could not extract preconditioner from optimizer state: {e}\n"
            f"Falling back to constant preconditioner (if learning_rate is provided)."
        )
        return None


def create_constant_preconditioner(
    model: nn.Module,
    lr: float,
    device: Optional[torch.device] = None
) -> DiagonalPreconditioner:
    """
    Create a constant diagonal preconditioner (for SGD or fixed learning rate).
    
    This creates a preconditioner P = (1/lr) * I, which is appropriate for
    SGD or when you want to compute eigenvalues of the preconditioned Hessian
    with a constant learning rate.
    
    Args:
        model: The model to get parameter count from
        lr: Learning rate
        device: Device for the preconditioner tensor (defaults to model's device)
        
    Returns:
        DiagonalPreconditioner with P = 1 / lr (constant diagonal)
        
    Example:
        >>> from implementations.utils.measure_checkpoint_min_snr import create_constant_preconditioner
        >>> P = create_constant_preconditioner(model, lr=2e-4)
        >>> lambda_max = compute_lambda_max_from_checkpoint(
        ...     checkpoint_path='./runs/checkpoint.pt',
        ...     model=model,
        ...     loss_fn=loss_fn,
        ...     data_sample=data_sample,
        ...     P=P
        ... )
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    if device is None:
        device = next(model.parameters()).device
    
    first_param = next(model.parameters())
    P_diag = torch.ones(n_params, device=device, dtype=first_param.dtype) / lr
    
    return DiagonalPreconditioner(P_diag)

