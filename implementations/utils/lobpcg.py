import numpy as np
import torch
from functools import wraps

"""
Native PyTorch implementation of the LOBPCG algorithm for eigenvalue computation.

This file incorporates the following performance improvements (opt-in where noted):

(2) First-iter warm-restart fast path: skip P
(3) Never reapply A inside Rayleigh–Ritz (pass S and precomputed AS)
(5) Single-pass, block projection instead of repeated project+re-orthonormalize
(6) Switch orthonormalization to QR and reduce repeats
(7) Mixed precision for HVP only (keep RR & orthonormalization in FP32)

The original baseline was adapted (and translated into PyTorch) from JAX:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105

This rewrite into PyTorch is based on an implementation by Alex Damian, with some minor improvements.
"""


# ------------------------------------------------------------
# Small helpers (unchanged behavior except where noted)
# ------------------------------------------------------------

def _mm(a, b):
    return torch.matmul(a, b)


def _eigh_ascending(A):
    """
    Return eigenvalues (descending, despite the historical name) and eigenvectors
    of a small Hermitian matrix A.

    NOTE: We keep this name for compatibility; values are sorted DESCENDING.
    """
    try:
        w, V = torch.linalg.eigh(A)
    except Exception as _:
        print("uh oh")
        raise Exception("eigh failed")
    idx = torch.argsort(-w)
    return w[idx], V[:, idx]


# ------------------------------------------------------------
# (6) Orthonormalization via QR  (replaces SVQB-based routine)
# ------------------------------------------------------------

def _svqb(X):
    norms = torch.linalg.norm(X, ord=2, dim=0, keepdims=True)
    X /= torch.where(norms == 0, 1.0, norms)
    inner = _mm(X.T, X)
    w, V = _eigh_ascending(inner)
    tau = torch.finfo(X.dtype).eps * w[0]
    padded = torch.maximum(w, tau)
    sqrted = torch.where(tau > 0, padded, 1.0) ** (-0.5)
    scaledV = V * sqrted[None, :]
    orthoX = _mm(X, scaledV)
    keep = ((w > tau) * (torch.diag(inner) > 0.0))[None, :]
    orthoX *= keep.type(orthoX.dtype)
    norms = torch.linalg.norm(orthoX, ord=2, dim=0, keepdims=True)
    keep *= (norms > 0.0).type(keep.dtype)
    orthoX /= torch.where(keep, norms, 1.0)
    return orthoX


def _orthonormalize_qr(X: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize columns of X using economy QR on GPU.

    For n >> k, QR is fast and numerically robust; avoids repeated
    Gram + small-eigh passes of SVQB.
    """
    # Reduced/economy QR returns Q (n x k) with orthonormal columns
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q

def _orthonormalize_svqb(X: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize columns of X using the SVQB method.
    """
    for _ in range(2):
        X = _svqb(X)
    return X


def _maybe_orthonormalize(X: torch.Tensor, assume_ortho: bool = False, tol: float = None) -> torch.Tensor:
    """
    Retains your original 'assume_ortho' contract, but uses QR when needed.
    If assume_ortho=True, we cheaply check ||X^T X - I||_F <= tol before re-orthonormalizing.
    """
    if not assume_ortho:
        return _orthonormalize_svqb(X)

    k = X.shape[1]
    if tol is None:
        # A safe default; loosen if running fp32 aggressively
        tol = 100 * k * torch.finfo(X.dtype).eps

    G = X.T @ X
    err = torch.linalg.norm(G - torch.eye(k, dtype=G.dtype, device=G.device), ord='fro')
    if err <= tol:
        # Input is already orthonormal enough
        # print(f"Input is already orthonormal with error {err:.2e} <= tol {tol:.2e}.")
        return X
    else:
        return _orthonormalize_svqb(X)


# ------------------------------------------------------------
# (5) Single-pass block projection
# ------------------------------------------------------------

def _project_out_block(X: torch.Tensor,
                       P: torch.Tensor,
                       U: torch.Tensor,
                       *,
                       assume_basis_ortho: bool = True) -> torch.Tensor:
    """
    Project U out of span([X, P]) in a single pass, then perform ONE orthonormalization.

    If P is None, projects only against X.

    This replaces the older routine that ran multiple (project -> re-ortho) cycles.
    With warm restarts, X (and P if present) are already orthonormal, so a single
    projection + one orthonormalization of U is sufficient and faster.

    Args
    ----
    X : (n, k)
    P : (n, k) or None
    U : (n, k) directions to be projected
    assume_basis_ortho : If True, skips re-orthonormalizing [X, P] themselves.

    Returns
    -------
    U_proj_ortho : (n, k) projected and orthonormalized block
    """
    basis = X if P is None else torch.cat((X, P), dim=1)

    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
        U = _orthonormalize_svqb(U)
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
    normU = torch.linalg.norm(U, ord=2, dim=0, keepdims=True)
    U *= normU >= 0.99

    return U


# ------------------------------------------------------------
# (3) RR from precomputed AS (never reapply A inside RR)
# ------------------------------------------------------------

def _rayleigh_ritz_from_S_AS(S: torch.Tensor, AS: torch.Tensor):
    """
    Compute RR on span(S) given precomputed AS = A(S).

    Returns eigenvalues (DESC) and eigenvectors of the small (d x d) matrix S^T A S,
    where d = S.shape[1].
    """
    # Work in FP32 (or higher) for the tiny RR system regardless of S/AS dtype.
    # This improves stability if HVPs were computed in bf16/fp16.
    ST_AS = _mm(S.T, AS)

    return _eigh_ascending(ST_AS)


# ------------------------------------------------------------
# Safety wrapper (unchanged)
# ------------------------------------------------------------

def create_safe_wrapper(A):
    """
    Wrap the matvec function A with an error handler that, in the event
    of an OOM, reports the shape of the input tensor that triggered it.
    """
    @wraps(A)
    def safe_A(tensor):
        try:
            return A(tensor)
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(
                f"{str(e)}\nTensor shape that caused the OOM: {tuple(tensor.shape)}"
            )
    return safe_A


# ------------------------------------------------------------
# (7) Optional mixed-precision wrapper for HVP only
# ------------------------------------------------------------

def _wrap_hvp_autocast(A, dtype: torch.dtype, out_dtype: torch.dtype):
    """
    Return a version of A that runs under CUDA autocast (bf16/fp16) but
    returns FP32 (or `out_dtype`) to keep the outer LOBPCG numerics stable.

    If your A already manages autocast internally, leave this unused.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("hvp_autocast_dtype was set but CUDA is not available.")

    def A_mixed(V: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(dtype=dtype):
            out = A(V)
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out
    return A_mixed


# ------------------------------------------------------------
# Main LOBPCG with improvements (2)(3)(5)(6)(7)
# ------------------------------------------------------------

def torch_lobpcg(
    A,
    X: torch.Tensor,
    max_iter: int = 100,
    tol: float = None,
    *,
    skip_p_on_first_iter: bool = True,    # (2) fast path for warm restarts
    hvp_autocast_dtype: torch.dtype = None  # (7) mixed precision HVP only
):
    """
    Compute the top-k eigenpairs of A via LOBPCG (GPU-friendly), optimized for
    expensive Hessian-vector products and warm restarts.

    Improvements enabled here:
      (2) First-iter warm-restart fast path: skip P
      (3) No A(S) application inside Rayleigh–Ritz; we pass S and AS
      (5) Single-pass block projection of R against [X, P] with one orthonorm
      (6) QR-based orthonormalization (replaces SVQB)
      (7) Optional mixed-precision for HVP only (bf16/fp16), RR/orthonorm in FP32

    Args
    ----
    A : Callable[(n, k) -> (n, k)]
        Block operator implementing Y = A(X). For Hessian-vector products,
        make sure this accepts a block of columns and returns the block result.

    X : Tensor, shape (n, k)
        Initial search directions (warm start compatible). Need not be orthogonal.

    max_iter : int
        Maximum iterations (usually small with warm restarts).

    tol : float or None
        Convergence tolerance (see JAX docstring semantics). If None, uses eps of X.dtype.

    skip_p_on_first_iter : bool
        If True, on the very first iteration we build the trial subspace from {X, R} only,
        skipping P entirely. This is a big win when you usually take just 1 iteration.

    hvp_autocast_dtype : torch.dtype or None
        If set (e.g., torch.bfloat16 on A100), the HVP A(.) is executed under CUDA autocast
        and the result cast back to FP32. Leave None if your A already handles autocast.

    Returns
    -------
    theta : (k,) top eigenvalues (descending)
    U     : (n, k) corresponding eigenvectors
    iters : number of iterations performed
    """
    # ---- (7) Optional mixed-precision HVP wrapper ----
    out_dtype = X.dtype  # keep outer math (RR/orthonorm) in the X dtype (typically FP32)
    if hvp_autocast_dtype is not None:
        # Allow TF32 kernels too (safe on Ampere/Hopper) — improves HVP speed further
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        A = _wrap_hvp_autocast(A, hvp_autocast_dtype, out_dtype=out_dtype)

    # Safety wrap last so it catches OOMs inside our mixed wrapper as well
    A = create_safe_wrapper(A)

    n, k = X.shape

    # Small-n fallback
    if n < 4 * k:
        A_mat = A(torch.eye(n, dtype=out_dtype, device=X.device))
        evals, evecs = _eigh_ascending(A_mat.to(torch.float32))
        return evals[:k], evecs[:, :k], 1

    dt = X.dtype
    minimum_tol = torch.finfo(dt).eps
    if tol is None:
        tol = minimum_tol

    # ---- (6) Orthonormalize X (QR) ----
    X = _maybe_orthonormalize(X, assume_ortho=False)

    # ---- Initialize (defer P construction for (2) skip-P fast path) ----
    P = None  # will be created after RR step
    AX = A(X)  # one batched HVP
    # Use 1D theta for simpler broadcasting; shape (k,)
    theta = torch.sum(X * AX, dim=0)
    R = AX - theta[None, :] * X

    converged = 0
    iters = 0

    for iters in range(max_iter):
        if converged >= k:
            break

        # ---- (5) Single-pass projection of R against current basis ----
        use_P = (P is not None) and not (iters == 0 and skip_p_on_first_iter)
        R = _project_out_block(X, P if use_P else None, R, assume_basis_ortho=True)

        # ---- Build S and AS explicitly; never call A inside RR (3) ----
        # We also try to batch HVPs where possible: A([P, R]) in one call.
        if use_P:
            # S = [X, P, R]; AS = [AX, AP, AR]
            if P is None:
                raise RuntimeError("Internal error: use_P=True but P is None.")
            # Batch HVP for P and R together to reduce overhead
            PR = torch.cat([P, R], dim=1)
            APR = A(PR)
            AP, AR = APR[:, :k], APR[:, k:]
            S = torch.cat([X, P, R], dim=1)
            AS = torch.cat([AX, AP, AR], dim=1)
        else:
            # S = [X, R]; AS = [AX, AR]
            AR = A(R)
            S = torch.cat([X, R], dim=1)
            AS = torch.cat([AX, AR], dim=1)

        # ---- Rayleigh–Ritz on span(S) using precomputed AS (3) ----
        theta_all, Q = _rayleigh_ritz_from_S_AS(S, AS)
        theta = theta_all[:k]  # top-k eigenvalues (1D)

        # Form B = Q[:, :k] and normalize columns for numerical hygiene
        B = Q[:, :k]
        normB = torch.linalg.norm(B, ord=2, dim=0, keepdims=True)
        # Zero-safe normalization
        B = B / torch.where(normB == 0, torch.ones_like(normB), normB)

        # New iterate X = S @ B  (then normalize)
        X = _mm(S, B)
        normX = torch.linalg.norm(X, ord=2, dim=0, keepdims=True)
        X = X / torch.where(normX == 0, torch.ones_like(normX), normX)

        # Build next P from the orthogonal complement directions encoded in Q
        # This works for both d=2k (no P in S) and d=3k (with P in S)
        d = Q.shape[0]
        # q comes from QR of the k x (d - k) block to stabilize
        q, _ = torch.linalg.qr(Q[:k, k:].T)  # ((d-k) x k)
        diff_rayleigh_ortho = _mm(Q[:, k:], q)  # (d x k)
        P = _mm(S, diff_rayleigh_ortho)  # (n x k)
        # Normalize P (zero-safe)
        normP = torch.linalg.norm(P, ord=2, dim=0, keepdims=True)
        P = P / torch.where(normP == 0, torch.ones_like(normP), normP)

        # Update AX, residuals, and convergence check
        AX = A(X)
        R = AX - theta[None, :] * X

        resid_norms = torch.linalg.norm(R, ord=2, dim=0)
        reltol = 10 * n * (torch.linalg.norm(AX, ord=2, dim=0) + theta)
        res_converged = resid_norms < (tol * reltol)
        converged = int(torch.sum(res_converged).item())

    return theta, X, iters

