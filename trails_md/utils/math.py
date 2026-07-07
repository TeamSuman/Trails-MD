import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Perform element-wise division with epsilon handling to avoid NaN/Inf."""
    denom_safe = np.where(np.abs(denominator) > eps, denominator, eps * np.sign(denominator + 1e-12))
    return numerator / denom_safe

def safe_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize array values to range [0, 1] safely."""
    denom = arr.max() - arr.min()
    if denom > eps:
        return (arr - arr.min()) / denom
    return np.ones_like(arr)

def safe_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along axis with overflow protection."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return safe_divide(exp_vals, np.sum(exp_vals, axis=axis, keepdims=True))


def encode_angles_sincos(angles: np.ndarray) -> np.ndarray:
    """Map angular features (radians) to their ``[sin θ, cos θ]`` embedding.

    Dihedral/torsion CVs are periodic: θ = +179° and θ = −179° are ~2° apart
    physically but nearly maximally far in raw radians, so feeding raw angles to
    Euclidean-distance methods (PCA/TICA/VAE/k-means, and grid/FPS/LOF spawners)
    silently corrupts the learned space and MSM microstates whenever an angle
    crosses ±π. The standard remedy is the sine/cosine embedding, which is
    continuous across the wrap. An ``(n_frames, k)`` angle array becomes
    ``(n_frames, 2k)`` with columns ``[sin θ_1..k, cos θ_1..k]``.
    """
    angles = np.asarray(angles, dtype=float)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
