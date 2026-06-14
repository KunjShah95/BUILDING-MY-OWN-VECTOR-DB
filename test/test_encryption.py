"""Tests for Encryption at Rest (Phase 9)."""

import os
import sys
from unittest.mock import patch

import pytest

from utils.encryption import EncryptionManager


class TestEncryptionManagerInit:
    def test_init_disabled_by_default(self):
        mgr = EncryptionManager()
        assert mgr.is_enabled is False

    def test_init_with_key(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        assert mgr.is_enabled is True

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ENCRYPTION_KEY", "env-key-32-bytes-long-for-testing!!")
        mgr = EncryptionManager()
        # Reload to pick up env var
        mgr = EncryptionManager()
        assert mgr.is_enabled is True


class TestEncryptionRoundTrip:
    def test_encrypt_decrypt_roundtrip(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        plaintext = b"hello vector database"
        encrypted = mgr.encrypt(plaintext)
        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)
        decrypted = mgr.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_decrypt_empty(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        assert mgr.decrypt(mgr.encrypt(b"")) == b""

    def test_encrypt_binary_data(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        data = bytes(range(256))
        assert mgr.decrypt(mgr.encrypt(data)) == data

    def test_encrypt_large_data(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        data = b"x" * 100000
        assert mgr.decrypt(mgr.encrypt(data)) == data

    def test_different_nonces_each_time(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        data = b"same data"
        e1 = mgr.encrypt(data)
        e2 = mgr.encrypt(data)
        assert e1 != e2  # nonces differ

    def test_different_keys_produce_different_ciphertext(self):
        data = b"secret"
        mgr1 = EncryptionManager(master_key=b"key-11111111111111111111111111111")
        mgr2 = EncryptionManager(master_key=b"key-22222222222222222222222222222")
        assert mgr1.encrypt(data) != mgr2.encrypt(data)

    def test_wrong_key_fails(self):
        mgr1 = EncryptionManager(master_key=b"correct-key-32-bytes-long-to-use!")
        mgr2 = EncryptionManager(master_key=b"wrong-key-32-bytes-long-to-use!!")
        encrypted = mgr1.encrypt(b"test")
        with pytest.raises(Exception):
            mgr2.decrypt(encrypted)


class TestEncryptionDisabled:
    def test_disabled_passthrough(self):
        mgr = EncryptionManager()
        data = b"plaintext"
        assert mgr.encrypt(data) == data
        assert mgr.decrypt(data) == data

    def test_disabled_file_ops_return_false(self, tmp_path):
        mgr = EncryptionManager()
        f = tmp_path / "test.dat"
        f.write_bytes(b"data")
        assert mgr.encrypt_file(str(f)) is False
        assert mgr.decrypt_file(str(f)) is False


class TestEncryptionFileOps:
    def test_encrypt_file_roundtrip(self, tmp_path):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        src = tmp_path / "plain.bin"
        src.write_bytes(b"file content for encryption test")
        enc = str(tmp_path / "encrypted.bin")

        assert mgr.encrypt_file(str(src), enc) is True
        enc_data = open(enc, "rb").read()
        assert enc_data != b"file content for encryption test"

        dec = str(tmp_path / "decrypted.bin")
        assert mgr.decrypt_file(enc, dec) is True
        assert open(dec, "rb").read() == b"file content for encryption test"

    def test_encrypt_file_in_place(self, tmp_path):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        f = tmp_path / "inplace.bin"
        f.write_bytes(b"data to encrypt in place")
        original = f.read_bytes()

        assert mgr.encrypt_file(str(f)) is True
        assert f.read_bytes() != original

        assert mgr.decrypt_file(str(f)) is True
        assert f.read_bytes() == original

    def test_encrypt_file_not_found(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        assert mgr.encrypt_file("/nonexistent/file.bin") is False
        assert mgr.decrypt_file("/nonexistent/file.bin") is False


class TestKeyDerivation:
    def test_derive_key_from_short_key(self):
        mgr = EncryptionManager(master_key=b"short")
        assert mgr.is_enabled is True
        data = b"test"
        assert mgr.decrypt(mgr.encrypt(data)) == data

    def test_derive_key_from_long_key(self):
        key = b"x" * 64
        mgr = EncryptionManager(master_key=key)
        assert mgr.is_enabled is True
        assert mgr.decrypt(mgr.encrypt(b"data")) == b"data"

    def test_same_key_derives_same_result(self, monkeypatch):
        """Two managers with same key should produce decryptable ciphertexts."""
        import hashlib
        mgr1 = EncryptionManager(master_key=b"same-source-key-32-bytes-long-here")
        mgr2 = EncryptionManager(master_key=b"same-source-key-32-bytes-long-here")
        ct = mgr1.encrypt(b"cross-manager test")
        assert mgr2.decrypt(ct) == b"cross-manager test"


class TestEncryptionMetadata:
    def test_encrypted_format_has_nonce_prefix(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        ct = mgr.encrypt(b"test")
        # First 12 bytes should be the nonce
        assert len(ct) > 12
        nonce = ct[:12]
        assert len(nonce) == 12  # AES-GCM standard nonce

    def test_encrypt_different_sizes(self):
        mgr = EncryptionManager(master_key=b"test-key-32-bytes-long-for-testing!")
        for size in [1, 16, 32, 100, 1000]:
            data = b"x" * size
            ct = mgr.encrypt(data)
            dec = mgr.decrypt(ct)
            assert dec == data, f"Failed for size {size}"
