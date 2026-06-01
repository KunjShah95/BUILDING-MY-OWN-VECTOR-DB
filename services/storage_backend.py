from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Union
import os
import io
import logging

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend for media and index files."""

    @abstractmethod
    def save(self, path: str, data: Union[bytes, BinaryIO]) -> bool:
        ...

    @abstractmethod
    def load(self, path: str) -> Optional[bytes]:
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list:
        ...


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage."""

    def save(self, path: str, data: Union[bytes, BinaryIO]) -> bool:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if isinstance(data, bytes):
            with open(path, "wb") as f:
                f.write(data)
        else:
            with open(path, "wb") as f:
                f.write(data.read())
        return True

    def load(self, path: str) -> Optional[bytes]:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def delete(self, path: str) -> bool:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_keys(self, prefix: str = "") -> list:
        if not os.path.exists(prefix):
            return []
        result = []
        for root, dirs, files in os.walk(prefix):
            for f in files:
                result.append(os.path.join(root, f))
        return result


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, bucket: str, region: str = "us-east-1",
                 endpoint_url: Optional[str] = None,
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None):
        self.bucket = bucket
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=region,
            )
            self.client = session.client("s3", endpoint_url=endpoint_url)
            try:
                self.client.head_bucket(Bucket=bucket)
            except Exception:
                self.client.create_bucket(Bucket=bucket)
        except ImportError:
            raise RuntimeError("boto3 required: pip install boto3")

    def save(self, path: str, data: Union[bytes, BinaryIO]) -> bool:
        if isinstance(data, bytes):
            self.client.put_object(Bucket=self.bucket, Key=path, Body=data)
        else:
            self.client.upload_fileobj(data, self.bucket, path)
        return True

    def load(self, path: str) -> Optional[bytes]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=path)
            return response["Body"].read()
        except Exception:
            return None

    def exists(self, path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

    def delete(self, path: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]


class AzureStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""

    def __init__(self, container: str,
                 connection_string: Optional[str] = None):
        self.container = container
        try:
            from azure.storage.blob import BlobServiceClient
            conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            self.service = BlobServiceClient.from_connection_string(conn_str)
            self.container_client = self.service.get_container_client(container)
            try:
                self.container_client.create_container()
            except Exception:
                pass
        except ImportError:
            raise RuntimeError("azure-storage-blob required: pip install azure-storage-blob")

    def save(self, path: str, data: Union[bytes, BinaryIO]) -> bool:
        blob = self.container_client.get_blob_client(path)
        if isinstance(data, bytes):
            blob.upload_blob(data, overwrite=True)
        else:
            blob.upload_blob(data, overwrite=True)
        return True

    def load(self, path: str) -> Optional[bytes]:
        try:
            blob = self.container_client.get_blob_client(path)
            return blob.download_blob().readall()
        except Exception:
            return None

    def exists(self, path: str) -> bool:
        try:
            blob = self.container_client.get_blob_client(path)
            blob.get_blob_properties()
            return True
        except Exception:
            return False

    def delete(self, path: str) -> bool:
        try:
            blob = self.container_client.get_blob_client(path)
            blob.delete_blob()
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list:
        return [b.name for b in self.container_client.list_blobs(name_starts_with=prefix)]


class StorageFactory:
    """Factory for creating storage backends based on config."""

    @staticmethod
    def create(provider: str = "local") -> StorageBackend:
        provider = (provider or "local").lower()
        if provider == "s3":
            return S3StorageBackend(
                bucket=os.getenv("S3_BUCKET", "vector-db-storage"),
                region=os.getenv("S3_REGION", "us-east-1"),
                endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            )
        elif provider == "azure":
            return AzureStorageBackend(
                container=os.getenv("AZURE_CONTAINER", "vector-db-storage"),
            )
        else:
            return LocalStorageBackend()


_storage_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Get (or create) the global storage backend."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = StorageFactory.create(os.getenv("STORAGE_PROVIDER", "local"))
    return _storage_backend
