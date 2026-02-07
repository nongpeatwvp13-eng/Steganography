from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import base64

class SecureAESCipher:
    """Secure AES-256-GCM encryption/decryption with PBKDF2 key derivation"""
    
    def __init__(self, password):
        self.salt = b'\x14\xeb\xfe\x01\x12\x8a\x91\xf2' 
        self.key = PBKDF2(password, self.salt, dkLen=32, count=100000)

    def encrypt(self, plaintext):
        """Encrypt plaintext using AES-256-GCM"""
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=get_random_bytes(12))
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
        return base64.b64encode(cipher.nonce + tag + ciphertext).decode('utf-8')

    def decrypt(self, b64_ciphertext):
        """Decrypt ciphertext using AES-256-GCM"""
        try:
            data = base64.b64decode(b64_ciphertext)
            nonce = data[:12]
            tag = data[12:28] 
            ciphertext = data[28:]
            
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
            return decrypted_data.decode('utf-8')
        except (ValueError, KeyError, UnicodeDecodeError) as e:
            return f"Error: Incorrect password or corrupted data - {str(e)}"
