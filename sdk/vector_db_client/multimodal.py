import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from vector_db_client._http import json_post, multipart_post
from vector_db_client.models import SearchResult

FileSource = Union[str, Path, bytes]


class MultimodalAPI:
    def __init__(self, client):
        self._client = client

    @staticmethod
    def _file_tuple(source: FileSource, field_name: str = "file"):
        if isinstance(source, bytes):
            return {field_name: ("upload.bin", source, "application/octet-stream")}
        path = Path(source)
        content = path.read_bytes()
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
        return {field_name: (path.name, content, mime)}

    def ingest_text(
        self,
        collection_id: str,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"text": text}
        if metadata:
            body["metadata"] = metadata
        if vector_id:
            body["vector_id"] = vector_id
        return json_post(
            self._client,
            f"/collections/{collection_id}/ingest/text",
            body,
        )

    def search_text(
        self,
        collection_id: str,
        query: str,
        *,
        k: int = 5,
        method: str = "brute",
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        body: Dict[str, Any] = {"query": query, "k": k, "method": method}
        if filters:
            body["filters"] = filters
        data = json_post(
            self._client,
            f"/collections/{collection_id}/search/text",
            body,
        )
        return SearchResult.from_api(data)

    def ingest_image(
        self,
        collection_id: str,
        *,
        path: Optional[FileSource] = None,
        data: Optional[bytes] = None,
        filename: str = "upload.jpg",
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = data if data is not None else path
        if source is None:
            raise ValueError("Provide path or data for image ingest")
        files = self._file_tuple(source if isinstance(source, bytes) else Path(source))
        form: Dict[str, Any] = {}
        if metadata:
            form["metadata"] = json.dumps(metadata)
        if vector_id:
            form["vector_id"] = vector_id
        return multipart_post(
            self._client,
            f"/collections/{collection_id}/ingest/image",
            files=files,
            data=form,
        )

    def ingest_audio(
        self,
        collection_id: str,
        *,
        path: Optional[FileSource] = None,
        data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = data if data is not None else path
        if source is None:
            raise ValueError("Provide path or data for audio ingest")
        files = self._file_tuple(source if isinstance(source, bytes) else Path(source))
        form: Dict[str, Any] = {}
        if metadata:
            form["metadata"] = json.dumps(metadata)
        if vector_id:
            form["vector_id"] = vector_id
        return multipart_post(
            self._client,
            f"/collections/{collection_id}/ingest/audio",
            files=files,
            data=form,
        )

    def search_image(
        self,
        collection_id: str,
        *,
        path: Optional[FileSource] = None,
        data: Optional[bytes] = None,
        k: int = 5,
        method: str = "brute",
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        source = data if data is not None else path
        if source is None:
            raise ValueError("Provide path or data for image search")
        files = self._file_tuple(source if isinstance(source, bytes) else Path(source))
        form: Dict[str, Any] = {"k": str(k), "method": method}
        if filters:
            form["filters"] = json.dumps(filters)
        payload = multipart_post(
            self._client,
            f"/collections/{collection_id}/search/image",
            files=files,
            data=form,
        )
        return SearchResult.from_api(payload)

    def search_audio(
        self,
        collection_id: str,
        *,
        path: Optional[FileSource] = None,
        data: Optional[bytes] = None,
        k: int = 5,
        method: str = "brute",
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        source = data if data is not None else path
        if source is None:
            raise ValueError("Provide path or data for audio search")
        files = self._file_tuple(source if isinstance(source, bytes) else Path(source))
        form: Dict[str, Any] = {"k": str(k), "method": method}
        if filters:
            form["filters"] = json.dumps(filters)
        payload = multipart_post(
            self._client,
            f"/collections/{collection_id}/search/audio",
            files=files,
            data=form,
        )
        return SearchResult.from_api(payload)


class AsyncVectorDBClient:
    """Optional async client stub for future expansion."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Async client is not implemented yet; use VectorDBClient")
