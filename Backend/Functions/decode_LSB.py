import numpy as np
from PIL import Image

from .AES_256 import SecureAESCipher
from .common import derive_seed, header_positions
from .decide import AdaptiveLSBCore


def decode_LSB(stego_path: str, password: str) -> str:
    img = Image.open(stego_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    flat      = img_array.reshape(-1)
    flat_size = flat.size

    seed          = derive_seed(password)
    hdr_positions = header_positions(flat_size, seed)

    header_bits        = flat[hdr_positions] & np.uint8(1)
    payload_bit_length = int(np.packbits(header_bits).view(">u4")[0])

    if payload_bit_length <= 0:
        return "Error: No hidden message"

    core          = AdaptiveLSBCore(seed_key=password, exclude_positions=hdr_positions)
    real_capacity = core.capacity(img_array)

    if payload_bit_length > real_capacity:
        return "Error: Invalid header — corrupted or wrong password"

    extracted_bits = core.decode(img_array, payload_bit_length)

    if len(extracted_bits) < payload_bit_length:
        return "Error: Corrupted data"

    payload_bytes = np.packbits(extracted_bits).tobytes()

    cipher = SecureAESCipher(password)
    return cipher.decrypt(payload_bytes)
