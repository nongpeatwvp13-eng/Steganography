import hashlib

import numpy as np

HEADER_BITS = 32


def derive_seed(password: str) -> int:
    """Derive a deterministic seed from a password using SHA-256."""
    d = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(d[:4], "big")


def header_positions(flat_size: int, seed: int) -> np.ndarray:
    """Generate sorted random header positions for embedding the payload length."""
    rng = np.random.Generator(np.random.PCG64(seed))
    pos = rng.choice(flat_size, size=HEADER_BITS, replace=False)
    pos.sort()
    return pos.astype(np.int64)
