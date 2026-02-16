from PIL import Image
import numpy as np
from .encode_LSB import encode_LSB
from .decode_LSB import decode_LSB
from .decide import initialize_embedding_map


def encode_message(input_img_path, message, password, output_img_path):
    return encode_LSB(input_img_path, message, password, output_img_path)


def decode_message(stego_img_path, password):
    return decode_LSB(stego_img_path, password)


def get_image_stats(img_path, real_capacity=False):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())
            file_format = img.format

            if real_capacity:
                img_array = np.array(img, dtype=np.uint8)

        total_pixels = width * height

        theoretical_max_bits = total_pixels * channels * 2

        delimiter_bits = len("######END######") * 8

        if real_capacity:
            embedding_map = initialize_embedding_map(img_array, max_bits=2)
            capacity_bits = int(np.sum(embedding_map))
            capacity_bits = max(0, capacity_bits - delimiter_bits)
            capacity_type = "real"
        else:
            # Fast heuristic estimate
            encryption_overhead = 0.3
            safety_margin = 0.5

            capacity_bits = int(
                (theoretical_max_bits - delimiter_bits)
                * (1 - encryption_overhead)
                * safety_margin
            )
            capacity_type = "estimated"

        max_capacity_chars = capacity_bits // 8

        return {
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "mode": mode,
            "format": str(file_format) if file_format else "Unknown",
            "total_pixels": int(total_pixels),
            "theoretical_max_bits": int(theoretical_max_bits),
            "practical_max_bits": int(capacity_bits),
            "max_capacity_chars": int(max_capacity_chars),
            "capacity_type": capacity_type,
            "notes": (
                "Real capacity matches adaptive embedding map"
                if real_capacity
                else "Estimated capacity with safety margin"
            ),
        }

    except Exception as e:
        return {
            "error": str(e),
            "width": 0,
            "height": 0,
            "channels": 0,
            "mode": "Unknown",
            "format": "Unknown",
            "total_pixels": 0,
            "theoretical_max_bits": 0,
            "practical_max_bits": 0,
            "max_capacity_chars": 0,
            "capacity_type": "error",
        }