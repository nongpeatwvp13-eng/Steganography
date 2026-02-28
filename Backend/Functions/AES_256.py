from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
KDF_ITERATIONS = 100_000


class SecureAESCipher:

    def __init__(self, password: str) -> None:
        self.password = password

    def _derive_key(self, salt: bytes) -> bytes:
        return PBKDF2(self.password, salt, dkLen=32, count=KDF_ITERATIONS)

    def encrypt(self, plaintext: str) -> bytes:
        salt = get_random_bytes(SALT_SIZE)
        key = self._derive_key(salt)
        cipher = AES.new(key, AES.MODE_GCM, nonce=get_random_bytes(NONCE_SIZE))
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
        # Format: salt + nonce + tag + ciphertext
        return salt + cipher.nonce + tag + ciphertext

    def decrypt(self, data: bytes) -> str:
        try:
            salt = data[:SALT_SIZE]
            nonce = data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
            tag = data[SALT_SIZE + NONCE_SIZE:SALT_SIZE + NONCE_SIZE + TAG_SIZE]
            ciphertext = data[SALT_SIZE + NONCE_SIZE + TAG_SIZE:]

            key = self._derive_key(salt)
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
            return decrypted_data.decode('utf-8')

        except (ValueError, KeyError, UnicodeDecodeError) as e:
            return f"Error: Incorrect password or corrupted data - {str(e)}"
