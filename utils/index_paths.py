import os
from typing import Optional


def get_index_dir(collection_id: Optional[str] = None) -> str:
    """Directory for index files; global indexes live at repo root."""
    if collection_id:
        return os.path.join("indexes", collection_id)
    return "."


def get_hnsw_path(collection_id: Optional[str] = None) -> str:
    if collection_id:
        return os.path.join(get_index_dir(collection_id), "hnsw_index_data.json")
    return "hnsw_index_data.json"


def get_ivf_path(collection_id: Optional[str] = None) -> str:
    if collection_id:
        return os.path.join(get_index_dir(collection_id), "ivf.json")
    return "ivf_index_data.json"



def get_pq_path(collection_id: Optional[str] = None) -> str:
    if collection_id:
        return os.path.join(get_index_dir(collection_id), "pq_index.json")
    return "pq_index_data.json"


def ensure_index_dir(collection_id: Optional[str] = None) -> str:
    directory = get_index_dir(collection_id)
    if collection_id:
        os.makedirs(directory, exist_ok=True)
    return directory
