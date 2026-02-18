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
    total_pixels         = rows * cols
    theoretical_max_bits = total_pixels * channels * 2

    if not real_capacity:
        capacity      = theoretical_max_bits // 2
        capacity_type = "estimated"
    else:
        core          = AdaptiveLSBCore()
        capacity      = core.capacity(arr)
        capacity_type = "real"

    return {
        "width":                int(cols),
        "height":               int(rows),
        "channels":             int(channels),
        "total_pixels":         int(total_pixels),
        "theoretical_max_bits": int(theoretical_max_bits),
        "practical_max_bits":   int(capacity),
        "max_capacity_chars":   int(capacity // 8),
        "capacity_type":        capacity_type,
    }