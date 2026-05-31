import json
from typing import Any, Dict, Optional

import httpx

from vector_db_client.exceptions import VectorDBHTTPError


def raise_for_status(response: httpx.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {"message": response.text}

    if response.is_success:
        return payload

    detail = payload.get("detail", payload)
    raise VectorDBHTTPError(response.status_code, detail)


def json_post(client, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = client.post(path, json=body or {})
    return raise_for_status(response)


def multipart_post(
    client,
    path: str,
    *,
    files: dict,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = client.post(path, files=files, data=data or {})
    return raise_for_status(response)
