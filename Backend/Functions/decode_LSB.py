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


def decode_LSB(stego_path, password):

    img = Image.open(stego_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    flat = img_array.reshape(-1)
    flat_size = flat.size

    seed = _seed(password)
    header_positions = _header_positions(flat_size, seed)

    header_bits = flat[header_positions] & 1

    payload_bit_length = int(
        np.packbits(header_bits).view(">u4")[0]
    )

    if payload_bit_length <= 0:
        return "Error: No hidden message"

    core = AdaptiveLSBCore()

    extracted_bits = core.decode(img_array, payload_bit_length)

    if len(extracted_bits) < payload_bit_length:
        return "Error: Corrupted data"

    payload_bytes = np.packbits(extracted_bits).tobytes()

    cipher = SecureAESCipher(password)
    return cipher.decrypt(payload_bytes)