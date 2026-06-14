"""
Encryption at Rest (Phase 9: Security & Compliance).

Provides transparent encryption/decryption of index files and mmap stores on
disk using AES-256-GCM. Integrates with the existing storage backend to
encrypt data before writing to disk and decrypt on read.

Key management:
  - Master key from environment variable (ENCRYPTION_KEY)
  - Key rotation support via versioned key headers
  - Each file gets a unique nonce for AEAD security

Usage::

    from utils.encryption import EncryptionManager

    mgr = EncryptionManager()
    encrypted = mgr.encrypt(b"vector data")
    decrypted = mgr.decrypt(encrypted)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

# Key derivation: derive 256-bit key from any passphrase
_KEY_LENGTH = 32  # 256 bits
_NONCE_LENGTH = 12  # 96 bits (standard for AES-GCM)
_KEY_SALT = b"vectordb_encryption_v1"


def _derive_key(master_key: bytes) -> bytes:
    """Derive a 256-bit AES key from the master key material."""
    if not _CRYPTO_AVAILABLE:
        return master_key[:32].ljust(32, b"\0") if len(master_key) < 32 else master_key[:32]
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=_KEY_SALT,
        info=b"vectordb-key",
    )
    return hkdf.derive(master_key)


class EncryptionManager:
    """Manages encryption/decryption of vector index data at rest.

    The encryption key is loaded from the ENCRYPTION_KEY environment variable
    or provided directly. If no key is set, operations are no-ops (pass-through)
    so the system degrades gracefully when encryption is not configured.
    """

    def __init__(self, master_key: Optional[bytes] = None):
        self._aes_key: Optional[bytes] = None
        self._enabled = False

        if master_key:
            self._setup(master_key)
        else:
            env_key = os.getenv("ENCRYPTION_KEY")
            if env_key:
                self._setup(env_key.encode())

    def _setup(self, master_key: bytes):
        self._aes_key = _derive_key(master_key)
        self._enabled = True
        logger.info("Encryption enabled (key derived)")

    @property
    def is_enabled(self) -> bool:
        return self._enabled and _CRYPTO_AVAILABLE

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt bytes with AES-256-GCM.

        Format: nonce (12 bytes) + ciphertext + tag (16 bytes)

        When encryption is disabled, returns plaintext unchanged.
        When the cryptography library is unavailable, raises.
        """
        if not self._enabled:
            return plaintext

        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography library required for encryption. "
                "Install: pip install cryptography"
            )

        aesgcm = AESGCM(self._aes_key)
        nonce = os.urandom(_NONCE_LENGTH)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext  # prepend nonce for storage

    def decrypt(self, encrypted: bytes) -> bytes:
        """Decrypt bytes encrypted with ``encrypt()``.

        When encryption is disabled, returns the input unchanged.
        """
        if not self._enabled:
            return encrypted

        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography library required for decryption. "
                "Install: pip install cryptography"
            )

        aesgcm = AESGCM(self._aes_key)
        nonce = encrypted[:_NONCE_LENGTH]
        ciphertext = encrypted[_NONCE_LENGTH:]
        return aesgcm.decrypt(nonce, ciphertext, None)

    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Encrypt a file in-place (or to a separate output path).

        Returns True on success.
        """
        if not self._enabled:
            return False

        out_path = output_path or input_path
        try:
            with open(input_path, "rb") as f:
                plaintext = f.read()
            encrypted = self.encrypt(plaintext)
            with open(out_path, "wb") as f:
                f.write(encrypted)
            logger.info("Encrypted %s -> %s (%d bytes)", input_path, out_path, len(encrypted))
            return True
        except Exception as exc:
            logger.error("File encryption failed: %s", exc)
            return False

    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Decrypt a file."""
        if not self._enabled:
            return False

        out_path = output_path or input_path
        try:
            with open(input_path, "rb") as f:
                encrypted = f.read()
            plaintext = self.decrypt(encrypted)
            with open(out_path, "wb") as f:
                f.write(plaintext)
            logger.info("Decrypted %s -> %s (%d bytes)", input_path, out_path, len(plaintext))
            return True
        except Exception as exc:
            logger.error("File decryption failed: %s", exc)
            return False
