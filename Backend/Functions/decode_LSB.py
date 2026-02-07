from PIL import Image
import numpy as np
from .AES_256 import SecureAESCipher

def get_pixel_complexity(img_array, i, j, channel):
    """Calculate local pixel complexity for adaptive bit extraction"""
    try:
        if i+2 > img_array.shape[0] or j+2 > img_array.shape[1]:
            return 0
        neighbors = img_array[i:i+2, j:j+2, channel].flatten()
        return np.std(neighbors)
    except:
        return 0

def decode_LSB(stego_img_path, password):
    """
    Decode message from steganographic image
    - Adaptive extraction matching encoding strategy
    - AES-256-GCM decryption
    """
    # Load image
    img = Image.open(stego_img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_array = np.array(img, dtype=np.uint8)
    rows, cols, channels = img_array.shape
    
    bin_data = ""
    max_bits = rows * cols * channels * 2
    delimiter = '######END######'

    # Extract bits
    for i in range(rows):
        for j in range(cols):
            for c in range(3):  # RGB channels
                if len(bin_data) >= max_bits:
                    break
                    
                # Match encoding complexity logic
                complexity = get_pixel_complexity(img_array, i, j, c)
                bits_to_extract = 2 if complexity > 20 else 1
                mask = (1 << bits_to_extract) - 1
                extracted_val = img_array[i, j, c] & mask
                bin_data += format(extracted_val, f'0{bits_to_extract}b')
                
                # Check for delimiter periodically (every 200 bytes = 1600 bits)
                if len(bin_data) >= 1600 and len(bin_data) % 8 == 0:
                    try:
                        temp_chars = ''.join(
                            chr(int(bin_data[b:b+8], 2)) 
                            for b in range(0, len(bin_data), 8) 
                            if len(bin_data[b:b+8]) == 8
                        )
                        
                        if delimiter in temp_chars:
                            encrypted_message = temp_chars.split(delimiter)[0]
                            cipher = SecureAESCipher(password)
                            decrypted_text = cipher.decrypt(encrypted_message)
                            
                            if decrypted_text.startswith("Error:"):
                                return "Error: Wrong password or corrupted data"
                            return decrypted_text
                    except (ValueError, UnicodeDecodeError):
                        continue
                        
            if len(bin_data) >= max_bits:
                break
        if len(bin_data) >= max_bits:
            break
    
    img.close()
    return "Error: No hidden message found or wrong password"
