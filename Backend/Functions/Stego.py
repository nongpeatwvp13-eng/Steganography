from PIL import Image
import numpy as np
from .encode_LSB import encode_LSB
from .decode_LSB import decode_LSB

def encode_message(input_img_path, message, password, output_img_path):
    """Wrapper for encoding message into image"""
    return encode_LSB(input_img_path, message, password, output_img_path)

def decode_message(stego_img_path, password):
    """Wrapper for decoding message from image"""
    return decode_LSB(stego_img_path, password)

def get_image_stats(img_path):
    """Get image statistics and capacity information"""
    try:
        img = Image.open(img_path)
        file_format = img.format
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img, dtype=np.uint8)
        rows, cols, channels = img_array.shape
        img.close()
        
        total_pixels = rows * cols
        # Theoretical max: 2 bits per channel * 3 channels per pixel
        theoretical_max_bits = total_pixels * channels * 2
        
        # Practical capacity accounting for:
        # - Delimiter: '######END######' = 15 chars = 120 bits
        # - AES overhead: ~20-30% for encryption + base64 encoding
        # - Safety margin: 50% for complex images with low-complexity regions
        delimiter_bits = len('######END######') * 8
        encryption_overhead = 0.3
        safety_margin = 0.5
        
        practical_max_bits = int(
            (theoretical_max_bits - delimiter_bits) * 
            (1 - encryption_overhead) * 
            safety_margin
        )
        practical_max_chars = practical_max_bits // 8
        
        return {
            'width': int(cols),
            'height': int(rows),
            'channels': int(channels),
            'total_pixels': int(total_pixels),
            'theoretical_max_bits': int(theoretical_max_bits),
            'practical_max_bits': int(practical_max_bits),
            'max_capacity_chars': int(practical_max_chars),
            'format': str(file_format) if file_format else 'Unknown',
            'mode': 'RGB',
            'overhead_info': 'Includes AES encryption + delimiter + safety margin'
        }
    except Exception as e:
        return {
            'error': str(e),
            'width': 0,
            'height': 0,
            'channels': 3,
            'total_pixels': 0,
            'theoretical_max_bits': 0,
            'practical_max_bits': 0,
            'max_capacity_chars': 0,
            'format': 'Unknown',
            'mode': 'RGB'
        }
