from PIL import Image
import numpy as np
import hashlib
from .AES_256 import SecureAESCipher
from .decide import initialize_embedding_map, cleanup


MASK_TABLE = {
    1: np.uint8(0b11111110),
    2: np.uint8(0b11111100),
}

HEADER_BITS = 32


def _get_prng_seed(password: str) -> int:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _get_header_positions(flat_size: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    positions = rng.choice(flat_size, size=HEADER_BITS, replace=False)
    positions.sort()
    return positions.astype(np.int64)


def encode_LSB(input_img_path, message, password, output_img_path):
    cipher = SecureAESCipher(password)
    encrypted_message = cipher.encrypt(message)
    payload = encrypted_message.encode("utf-8")

    msg_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    total_bits = int(msg_bits.size)

    img = Image.open(input_img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img, dtype=np.uint8)
    rows, cols, channels = img_array.shape

    print(f"Image size: {rows}x{cols} ({rows*cols:,} pixels)")
    print(f"Message bits (encrypted): {total_bits}")

    flat_size = rows * cols * channels

    seed = _get_prng_seed(password)
    header_positions = _get_header_positions(flat_size, seed)

    flat_img = img_array.ravel() 

    header_bits = np.unpackbits(
        np.array([total_bits], dtype=">u4").view(np.uint8)
    ) 

    for i, pos in enumerate(header_positions):
        flat_img[pos] = (flat_img[pos] & np.uint8(0b11111110)) | header_bits[i]

    embedding_map = initialize_embedding_map(img_array, max_bits=2)
    embedding_map = embedding_map.astype(np.uint8, copy=False)

    real_capacity = int(np.sum(embedding_map))
    if total_bits > real_capacity:
        cleanup()
        raise ValueError(
            f"Message too long! {total_bits} bits, capacity {real_capacity} bits"
        )

    flat_map = embedding_map.ravel()
    target_indices = np.nonzero(flat_map)[0]

    header_set = set(header_positions.tolist())
    target_indices = target_indices[
        ~np.isin(target_indices, list(header_set), assume_unique=True)
    ]

    print(f"Embedding into {target_indices.size:,} available channels (excl. header)")

    bit_idx = 0
    embedded_channels = 0
    bits1 = 0
    bits2 = 0

    for pos in target_indices:
        if bit_idx >= total_bits:
            break

        bits_to_embed = int(flat_map[pos])
        remaining = total_bits - bit_idx
        if bits_to_embed > remaining:
            bits_to_embed = remaining

        chunk = msg_bits[bit_idx:bit_idx + bits_to_embed]
        if bits_to_embed == 1:
            val = chunk[0]
        else:
            val = (chunk[0] << 1) | chunk[1]

        flat_img[pos] = (flat_img[pos] & MASK_TABLE[bits_to_embed]) | val

        bit_idx += bits_to_embed
        embedded_channels += 1

        if bits_to_embed == 1:
            bits1 += 1
        else:
            bits2 += 1

    if bit_idx < total_bits:
        cleanup()
        raise ValueError(
            f"Image capacity insufficient! Embedded {bit_idx}/{total_bits} bits"
        )

    print(f"Embedded {bit_idx} bits in {embedded_channels:,} channels")

    Image.fromarray(img_array).save(
        output_img_path,
        format="PNG",
        compress_level=1
    )

    cleanup()

    stats = {
        "image_size": (rows, cols),
        "total_pixels": rows * cols,
        "message_bits": total_bits,
        "capacity_bits": real_capacity,
        "capacity_used_percent": (total_bits / real_capacity) * 100,
        "channels_used": embedded_channels,
        "bits_embedded": {1: bits1, 2: bits2},
        "original_message_length": len(message),
        "encrypted_message_length": len(encrypted_message),
        "output_path": output_img_path,
    }

    return stats