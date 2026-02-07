from PIL import Image
import numpy as np
from .AES_256 import SecureAESCipher

def get_pixel_complexity(img_array, i, j, channel):
    """Calculate local pixel complexity for adaptive bit embedding"""
    try:
        if i+2 > img_array.shape[0] or j+2 > img_array.shape[1]:
            return 0
        neighbors = img_array[i:i+2, j:j+2, channel].flatten()
        return np.std(neighbors)
    except:
        return 0

def encode_LSB(input_img_path, message, password, output_img_path):
    """
    Encode message into image using adaptive LSB steganography
    - Uses 1-2 bits per channel based on pixel complexity
    - AES-256-GCM encryption for message security
    """
    # Encrypt message
    cipher = SecureAESCipher(password)
    encrypted_message = cipher.encrypt(message)
    message_with_delimiter = encrypted_message + '######END######'
    
    # Convert to binary
    bin_msg = ''.join(format(ord(char), '08b') for char in message_with_delimiter)
    
    # Load and prepare image
    img = Image.open(input_img_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img, dtype=np.uint8).copy()  # Make writable copy
    rows, cols, channels = img_array.shape
    
    # Check capacity
    max_capacity = rows * cols * channels * 2
    if len(bin_msg) > max_capacity:
        raise ValueError(f'Message too long! Message: {len(bin_msg)} bits, Capacity: {max_capacity} bits')
    
    # Statistics tracking
    stats = {
        'total_pixels': rows * cols * channels,
        'message_bits': len(bin_msg),
        'bits_embedded_1': 0,
        'bits_embedded_2': 0,
        'original_message_length': len(message),
        'encrypted_message_length': len(encrypted_message)
    }

    # Embed message
    idx = 0
    for i in range(rows):
        for j in range(cols):
            for c in range(3):  # RGB channels
                if idx >= len(bin_msg):
                    break
                    
                # Determine bits to embed based on complexity
                complexity = get_pixel_complexity(img_array, i, j, c)
                bits_to_embed = 2 if complexity > 20 else 1
                
                # Track statistics
                if bits_to_embed == 2:
                    stats['bits_embedded_2'] += 2
                else:
                    stats['bits_embedded_1'] += 1
                
                # Extract chunk and embed
                chunk = bin_msg[idx : idx + bits_to_embed]
                val = int(chunk, 2)
                mask = (0xFF << len(chunk)) & 0xFF
                img_array[i, j, c] = (img_array[i, j, c] & mask) | val
                idx += len(chunk)
                
            if idx >= len(bin_msg): 
                break
        if idx >= len(bin_msg): 
            break

    # Save encoded image as PNG (lossless)
    encoded_img = Image.fromarray(img_array, mode='RGB')
    encoded_img.save(output_img_path, format="PNG", compress_level=6)
    encoded_img.close()
    img.close()
    
    stats['capacity_used_percent'] = (len(bin_msg) / max_capacity) * 100
    return stats
