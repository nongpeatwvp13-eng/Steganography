import numpy as np
import hashlib
from PIL import Image
from .AES_256 import SecureAESCipher
from .decide import AdaptiveLSBCore

HEADER_BITS = 32


def _seed(password: str) -> int:
    d = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(d[:4], "big")


def encode_LSB(image_path, plaintext, password, output_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    flat_size = img_array.size
    if flat_size < HEADER_BITS:
        raise ValueError("Image too small for header")

    seed = _seed(password)
    header_positions = np.random.Generator(
        np.random.PCG64(seed)
    ).choice(flat_size, size=HEADER_BITS, replace=False)
    header_positions.sort()

    cipher        = SecureAESCipher(password)
    payload_bytes = cipher.encrypt(plaintext)
    payload_bits  = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    payload_bit_length = len(payload_bits)

    if payload_bit_length <= 0:
        raise ValueError("Empty payload")

    core      = AdaptiveLSBCore(seed_key=password, exclude_positions=header_positions)
    available = core.capacity(img_array)

    if payload_bit_length > available:
        raise ValueError(
            f"Message too large. Need {payload_bit_length} bits, "
            f"image capacity {available} bits."
        )

    img_array = core.encode(img_array, payload_bits)

    flat = img_array.reshape(-1)
    header_bits_arr = np.unpackbits(
        np.array([payload_bit_length], dtype=">u4").view(np.uint8)
    )
    flat[header_positions] = (flat[header_positions] & np.uint8(0xFE)) | header_bits_arr

    Image.fromarray(img_array).save(output_path)