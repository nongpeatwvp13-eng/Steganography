from .encode_LSB import encode_LSB
from .decode_LSB import decode_LSB
from .decide import AdaptiveLSBCore

from PIL import Image
import numpy as np


def encode_message(input_img_path, message, password, output_img_path):
    return encode_LSB(input_img_path, message, password, output_img_path)


def decode_message(stego_img_path, password):
    return decode_LSB(stego_img_path, password)


def get_image_stats(img_path, real_capacity=False):

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)

    rows, cols, channels = arr.shape
    total_pixels = rows * cols
    theoretical_max_bits = total_pixels * channels * 2

    if not real_capacity:
        capacity = theoretical_max_bits // 2
        capacity_type = "estimated"
    else:
        core = AdaptiveLSBCore()
        capacity = 0

        for rs in range(0, rows, core.block_rows):
            re = min(rs + core.block_rows, rows)
            block = arr[rs:re]
            zone = core._zone(block)

            for c in range(channels):
                channel = block[:, :, c]

                var = core._variance(channel)
                grad = core._grad(channel)
                score = core._score(zone, var, grad, c)

                mask = score >= 40
                local_idx = np.flatnonzero(mask.ravel())
                if local_idx.size == 0:
                    continue

                bits_per = np.where(score.ravel()[local_idx] >= 60, 2, 1)
                capacity += int(np.sum(bits_per))

                del var
                del grad
                del score
                del mask

            del block
            del zone

        capacity_type = "real"

    return {
        "width": int(cols),
        "height": int(rows),
        "channels": int(channels),
        "total_pixels": int(total_pixels),
        "theoretical_max_bits": int(theoretical_max_bits),
        "practical_max_bits": int(capacity),
        "max_capacity_chars": int(capacity // 8),
        "capacity_type": capacity_type,
    }