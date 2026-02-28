"""Tests for the AES-256 encryption module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from Functions.AES_256 import SecureAESCipher


def test_encrypt_decrypt_roundtrip():
    """Encrypting then decrypting should return the original message."""
    cipher = SecureAESCipher("testpassword1234")
    original = "Hello, World! This is a secret message."
    encrypted = cipher.encrypt(original)
    decrypted = cipher.decrypt(encrypted)
    assert decrypted == original


def test_different_passwords_fail():
    """Decrypting with a wrong password should return an error string."""
    cipher_enc = SecureAESCipher("correct_password")
    cipher_dec = SecureAESCipher("wrong_password")
    encrypted = cipher_enc.encrypt("secret data")
    result = cipher_dec.decrypt(encrypted)
    assert result.startswith("Error:")


def test_random_salt_produces_different_ciphertexts():
    """Two encryptions of the same plaintext with the same password should differ."""
    cipher = SecureAESCipher("samepassword")
    ct1 = cipher.encrypt("same message")
    ct2 = cipher.encrypt("same message")
    # Because salt and nonce are random, ciphertexts should differ
    assert ct1 != ct2


def test_empty_message():
    """Empty string should encrypt and decrypt correctly."""
    cipher = SecureAESCipher("password")
    encrypted = cipher.encrypt("")
    decrypted = cipher.decrypt(encrypted)
    assert decrypted == ""


def test_unicode_message():
    """Unicode characters should survive the round trip."""
    cipher = SecureAESCipher("unicodepass")
    original = "Hello, World!"
    encrypted = cipher.encrypt(original)
    decrypted = cipher.decrypt(encrypted)
    assert decrypted == original
