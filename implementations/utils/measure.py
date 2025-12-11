"""
Hessian matrix approximation and eigenvalue computation utilities.
Adapted from edge-of-stochastic-stability/utils/measure.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Optional, Tuple, Union
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


def create_hessian_vector_product(loss, net):
    """
    Create a Hessian-vector product function for use with LOBPCG.
    
    This function creates a closure that computes the Hessian-vector product
    H @ v where H is the Hessian of the loss function with respect to network parameters.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        
    Returns:
        callable: A function that takes a vector v and returns H @ v
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)
    
    def hessian_vector_product(v):
        """
        Compute Hessian-vector product H @ v.
        
        Args:
            v (Tensor): Vector(s) to multiply with Hessian. Can be 1D or 2D (for multiple vectors).
            
        Returns:
            Tensor: H @ v (same shape as v)
        """
        # Handle both 1D and 2D inputs for compatibility with LOBPCG
        if v.dim() == 1:
            # Single vector case
            grad_v = torch.dot(grads_vector, v)
            Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
            return flatt(Hv)
        elif v.dim() == 2:
            # Multiple vectors case (for LOBPCG)
            results = []
            for i in range(v.shape[1]):
                vi = v[:, i]
                grad_v = torch.dot(grads_vector, vi)
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                results.append(flatt(Hv))
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")
    
    return hessian_vector_product


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """
    Compute multiple eigenvalues of the Hessian using LOBPCG algorithm.
    
    This function computes the top-k eigenvalues of the Hessian matrix using the
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) algorithm.
    
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
    
    # Create Hessian-vector product function
    hessian_matvec = create_hessian_vector_product(loss, net)
    
    # Initialize vectors with priority: init_vectors > cached vectors > random
    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k:
            raise ValueError(f"init_vectors must have shape [n_params, {k}], got {X.shape}")
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
        else:
            X = torch.randn(n_params, k, device=device)
    else:
        # Use random initialization
        X = torch.randn(n_params, k, device=device)
    
    # Ensure X is on the correct device and has the right shape
    X = X.to(device)
    if X.shape != (n_params, k):
        X = X.reshape(n_params, k)
    
    # Run LOBPCG
    tol = reltol / (20 * n_params)  # Adjust tolerance based on problem size

    eigenvalues, eigenvectors, iterations = torch_lobpcg(
        hessian_matvec, X, max_iter=max_iterations, tol=tol
    )
    
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
                        use_power_iteration: bool = False):
    """
    Computes the top-k eigenvalues of the Hessian of the loss function at the current point.
    
    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.

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
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")
    
    device = next(net.parameters()).device

    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        # Use the existing power iteration implementation
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vectors, batched,
            eigenvector_cache, return_eigenvectors
        )
    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors, 
            eigenvector_cache, return_eigenvectors=True
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

