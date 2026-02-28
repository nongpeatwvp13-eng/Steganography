"""Tests for the full steganography encode/decode round-trip."""
import sys
import os
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from Functions.Stego import encode_message, decode_message, get_image_stats


def _create_test_image(width: int = 256, height: int = 256) -> str:
    """Create a random RGB test image and return its path."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = os.path.join(tempfile.mkdtemp(), "test_image.png")
    Image.fromarray(data).save(path)
    return path


def test_encode_decode_roundtrip():
    """Encoding then decoding should recover the original message."""
    img_path = _create_test_image()
    output_path = img_path.replace("test_image.png", "encoded.png")

    message = "Hello, steganography!"
    password = "test1234"

    encode_message(img_path, message, password, output_path)
    result = decode_message(output_path, password)

    assert result == message


def test_wrong_password_fails():
    """Decoding with the wrong password should return an error."""
    img_path = _create_test_image()
    output_path = img_path.replace("test_image.png", "encoded.png")

    encode_message(img_path, "secret", "correct_pass", output_path)
    result = decode_message(output_path, "wrong_pass")

    assert isinstance(result, str)
    assert "Error" in result


def test_get_image_stats_estimated():
    """get_image_stats should return valid capacity info."""
    img_path = _create_test_image(128, 128)
    stats = get_image_stats(img_path, real_capacity=False)

    assert stats["width"] == 128
    assert stats["height"] == 128
    assert stats["channels"] == 3
    assert stats["total_pixels"] == 128 * 128
    assert stats["max_capacity_chars"] > 0
    assert stats["capacity_type"] == "estimated"


def test_get_image_stats_real():
    """get_image_stats with real_capacity should use the adaptive core."""
    img_path = _create_test_image(64, 64)
    stats = get_image_stats(img_path, real_capacity=True)

    assert stats["capacity_type"] == "real"
    assert stats["practical_max_bits"] > 0


def test_message_too_large():
    """Encoding a message larger than capacity should raise ValueError."""
    img_path = _create_test_image(16, 16)  # Very small image
    output_path = img_path.replace("test_image.png", "encoded.png")

    long_message = "A" * 100000  # Way too long for a 16x16 image
    try:
        encode_message(img_path, long_message, "pass1234", output_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "too large" in str(e).lower() or "capacity" in str(e).lower()
