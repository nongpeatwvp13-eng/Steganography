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

def decode_LSB(stego_img_path, password):
    img = Image.open(stego_img_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    flat = img_array.ravel()
    flat_size = flat.size

    seed = _seed(password)
    header_positions = _header_positions(flat_size, seed)

    header_bits = np.zeros(HEADER_BITS, dtype=np.uint8)
    for i, pos in enumerate(header_positions):
        header_bits[i] = flat[pos] & 1

    payload_bit_length = int(
        np.packbits(header_bits).view(">u4")[0]
    )

    if payload_bit_length <= 0 or payload_bit_length > flat_size:
        return "Error: No hidden message"

    header_set = set(header_positions.tolist())
    decider = AdaptiveLSBDecider(2)
    stream = decider.stream(img_array)

    extracted_bits = np.zeros(payload_bit_length, dtype=np.uint8)
    bit_idx = 0
    rows, cols, channels = img_array.shape

    for i, j, c, bits in stream:
        if bit_idx >= payload_bit_length:
            break

        flat_index = (i * cols * channels) + (j * channels) + c

        if flat_index in header_set:
            continue

        if bits == 0:
            continue

        remaining = payload_bit_length - bit_idx
        use = min(bits, remaining)
        val = flat[flat_index]

        if use == 1:
            extracted_bits[bit_idx] = val & 1
        else:
            extracted_bits[bit_idx] = (val >> 1) & 1
            extracted_bits[bit_idx + 1] = val & 1

        bit_idx += use

    if bit_idx < payload_bit_length:
        return "Error: Corrupted data"

    byte_array = np.packbits(extracted_bits).tobytes()

    cipher = SecureAESCipher(password)
    try:
        return cipher.decrypt(byte_array)
    except Exception as e:
        return f"Error: {str(e)}"