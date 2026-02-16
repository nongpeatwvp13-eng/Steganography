import numpy as np
from PIL import Image
import hashlib
from .AES_256 import SecureAESCipher
from .decide import initialize_embedding_map, cleanup

HEADER_BITS = 32

def _get_prng_seed(password: str) -> int:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _get_header_positions(flat_size: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    positions = rng.choice(flat_size, size=HEADER_BITS, replace=False)
    positions.sort()
    return positions.astype(np.int64)


def decode_LSB(stego_img_path, password):
    with Image.open(stego_img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img, dtype=np.uint8)

    flat_img = img_array.ravel()
    flat_size = flat_img.size

    seed = _get_prng_seed(password)
    header_positions = _get_header_positions(flat_size, seed)

    header_bits = np.empty(HEADER_BITS, dtype=np.uint8)
    for i, pos in enumerate(header_positions):
        header_bits[i] = flat_img[pos] & np.uint8(1)

    payload_bit_length = int(
        np.packbits(header_bits).view(">u4")[0]
    )

    if payload_bit_length == 0 or payload_bit_length > flat_size:
        cleanup()
        return "Error: No hidden message found (invalid header)"

    print(f"Header decoded: expecting {payload_bit_length} payload bits")

    embedding_map = initialize_embedding_map(img_array, max_bits=2)
    embedding_map = embedding_map.astype(np.uint8, copy=False)

    flat_map = embedding_map.ravel()
    target_indices = np.nonzero(flat_map)[0]

    header_set = set(header_positions.tolist())
    target_indices = target_indices[
        ~np.isin(target_indices, list(header_set), assume_unique=True)
    ]

    print(f"Extracting from {target_indices.size:,} channels")

    extracted_bits = np.empty(payload_bit_length, dtype=np.uint8)
    bit_idx = 0

    for count, pos in enumerate(target_indices):
        if bit_idx >= payload_bit_length:
            break

        bits = int(flat_map[pos])
        val = flat_img[pos]

        remaining = payload_bit_length - bit_idx

        if bits == 1 or remaining == 1:
            extracted_bits[bit_idx] = val & 1
            bit_idx += 1
        else:
            extracted_bits[bit_idx]     = (val >> 1) & 1
            extracted_bits[bit_idx + 1] = val & 1
            bit_idx += 2

        if count % 1_000_000 == 0 and count > 0:
            print(f"  Processed {count:,} channels", end="\r")

    if bit_idx < payload_bit_length:
        cleanup()
        return f"Error: Could only extract {bit_idx}/{payload_bit_length} bits — image may be corrupted"

    num_bytes = payload_bit_length // 8
    extracted_bits = extracted_bits[:num_bytes * 8]
    byte_array = np.packbits(extracted_bits).tobytes()

    del extracted_bits
    del flat_map
    del target_indices
    del embedding_map
    cleanup()

    cipher = SecureAESCipher(password)
    try:
        decrypted = cipher.decrypt(byte_array.decode("utf-8"))
        return decrypted
    except Exception as e:
        return f"Error: Wrong password or corrupted data — {str(e)}"