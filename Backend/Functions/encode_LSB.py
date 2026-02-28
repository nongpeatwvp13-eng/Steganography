import numpy as np
from PIL import Image

from .AES_256 import SecureAESCipher
from .common import HEADER_BITS, derive_seed, header_positions
from .decide import AdaptiveLSBCore


def encode_LSB(
    image_path: str,
    plaintext: str,
    password: str,
    output_path: str,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    flat_size = img_array.size
    if flat_size < HEADER_BITS:
        raise ValueError("Image too small for header")

    seed = derive_seed(password)
    hdr_positions = header_positions(flat_size, seed)

    cipher        = SecureAESCipher(password)
    payload_bytes = cipher.encrypt(plaintext)
    payload_bits  = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    payload_bit_length = len(payload_bits)

    if payload_bit_length <= 0:
        raise ValueError("Empty payload")

    core      = AdaptiveLSBCore(seed_key=password, exclude_positions=hdr_positions)
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
    flat[hdr_positions] = (flat[hdr_positions] & np.uint8(0xFE)) | header_bits_arr

    Image.fromarray(img_array).save(output_path)
