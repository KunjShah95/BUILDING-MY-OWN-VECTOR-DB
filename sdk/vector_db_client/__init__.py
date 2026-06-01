from vector_db_client.client import VectorDBClient
from vector_db_client.exceptions import VectorDBError, VectorDBHTTPError

try:
    from vector_db_client.langchain_vectorstore import VectorDBVectorStore  # noqa: F401
except ImportError:
    # langchain-core not installed — LangChain integration unavailable
    pass

__all__ = [
    "VectorDBClient",
    "VectorDBError",
    "VectorDBHTTPError",
    "VectorDBVectorStore",
]
