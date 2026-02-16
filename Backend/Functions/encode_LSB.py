from PIL import Image
import numpy as np
import hashlib
from .AES_256 import SecureAESCipher
from .decide import AdaptiveLSBDecider

HEADER_BITS = 32

def _seed(password: str) -> int:
    d = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(d[:4], "big")

def _header_positions(flat_size: int, seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    pos = rng.choice(flat_size, size=HEADER_BITS, replace=False)
    pos.sort()
    return pos.astype(np.int64)

def encode_LSB(input_img_path, message, password, output_img_path):
    cipher = SecureAESCipher(password)
    encrypted = cipher.encrypt(message)

    if isinstance(encrypted, str):
        payload = encrypted.encode("utf-8")
    else:
        payload = encrypted

    msg_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    total_bits = int(msg_bits.size)

    img = Image.open(input_img_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    rows, cols, channels = img_array.shape
    flat = img_array.ravel()
    flat_size = flat.size

    seed = _seed(password)
    header_positions = _header_positions(flat_size, seed)

    header_bits = np.unpackbits(
        np.array([total_bits], dtype=">u4").view(np.uint8)
    )

    for i, pos in enumerate(header_positions):
        flat[pos] = (flat[pos] & 0b11111110) | header_bits[i]

    header_set = set(header_positions.tolist())
    decider = AdaptiveLSBDecider(2)
    stream = decider.stream(img_array)

    bit_idx = 0

    for i, j, c, bits in stream:
        if bit_idx >= total_bits:
            break

        flat_index = (i * cols * channels) + (j * channels) + c

        if flat_index in header_set:
            continue

        if bits == 0:
            continue

        remaining = total_bits - bit_idx
        use = min(bits, remaining)

        if use == 1:
            val = msg_bits[bit_idx]
        else:
            val = (msg_bits[bit_idx] << 1) | msg_bits[bit_idx + 1]

        mask = 0b11111110 if use == 1 else 0b11111100
        flat[flat_index] = (flat[flat_index] & mask) | val

        bit_idx += use

    if bit_idx < total_bits:
        raise ValueError("Image capacity insufficient")

    Image.fromarray(img_array).save(
        output_img_path,
        format="PNG",
        compress_level=1
    )