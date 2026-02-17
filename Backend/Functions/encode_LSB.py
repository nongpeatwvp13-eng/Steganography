import numpy as np
import hashlib
from PIL import Image
from .AES_256 import SecureAESCipher
from .decide import AdaptiveLSBCore


HEADER_BITS = 32


def _seed(password: str) -> int:
    d = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(d[:4], "big")


def _header_positions(flat_size: int, seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    pos = rng.choice(flat_size, size=HEADER_BITS, replace=False)
    pos.sort()
    return pos.astype(np.int64)


def encode_LSB(image_path, plaintext, password, output_path):

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    cipher = SecureAESCipher(password)
    payload_bytes = cipher.encrypt(plaintext)

    payload_bits = np.unpackbits(
        np.frombuffer(payload_bytes, dtype=np.uint8)
    )

    payload_bit_length = len(payload_bits)

    if payload_bit_length <= 0:
        raise ValueError("Empty payload")

    flat = img_array.reshape(-1)
    flat_size = flat.size

    seed = _seed(password)
    header_positions = _header_positions(flat_size, seed)

    header_bits = np.unpackbits(
        np.array([payload_bit_length], dtype=">u4").view(np.uint8)
    )

    for i, pos in enumerate(header_positions):
        flat[pos] &= 0b11111110
        flat[pos] |= header_bits[i]

    core = AdaptiveLSBCore()

    img_array = core.encode(img_array, payload_bits)

    if getattr(core, "last_bit_index", 0) < payload_bit_length:
        raise ValueError("Payload too large for image capacity")

    Image.fromarray(img_array).save(output_path)