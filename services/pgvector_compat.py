"""
Minimal PostgreSQL wire protocol server (text format) that proxies SQL queries
to the VectorService.

Supported SQL patterns:
  SELECT * FROM vectors WHERE collection='X' ORDER BY embedding <-> '[...]' LIMIT k
  INSERT INTO vectors (collection, embedding, metadata) VALUES (...)
  SELECT count(*) FROM vectors WHERE collection='X'

Usage:
    server = PgWireServer(vector_service)
    await server.start("0.0.0.0", 5433)
    ...
    await server.stop()
"""
import asyncio
import json
import logging
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostgreSQL wire protocol helpers (text format)
# ---------------------------------------------------------------------------

def _pack_int32(v: int) -> bytes:
    return struct.pack("!i", v)


def _pack_int16(v: int) -> bytes:
    return struct.pack("!h", v)


def _encode_str(s: str) -> bytes:
    return s.encode("utf-8") + b"\x00"


def _build_auth_ok() -> bytes:
    # AuthenticationOk: 'R' + len(8) + 0
    body = _pack_int32(0)
    return b"R" + _pack_int32(4 + len(body)) + body


def _build_ready_for_query() -> bytes:
    # ReadyForQuery: 'Z' + len(5) + 'I'
    return b"Z" + _pack_int32(5) + b"I"


def _build_error_response(msg: str) -> bytes:
    # ErrorResponse: 'E' + len + fields
    body = b"S" + _encode_str("ERROR")
    body += b"M" + _encode_str(msg)
    body += b"\x00"
    return b"E" + _pack_int32(4 + len(body)) + body


def _build_command_complete(tag: str) -> bytes:
    body = _encode_str(tag)
    return b"C" + _pack_int32(4 + len(body)) + body


def _build_row_description(columns: List[str]) -> bytes:
    """Build RowDescription message."""
    n = len(columns)
    body = _pack_int16(n)
    for col in columns:
        body += _encode_str(col)
        body += _pack_int32(0)   # table OID
        body += _pack_int16(0)   # column attr num
        body += _pack_int32(25)  # type OID: text
        body += _pack_int16(-1)  # type size
        body += _pack_int32(-1)  # type modifier
        body += _pack_int16(0)   # format: text
    return b"T" + _pack_int32(4 + len(body)) + body


def _build_data_row(values: List[Optional[str]]) -> bytes:
    """Build DataRow message."""
    n = len(values)
    body = _pack_int16(n)
    for v in values:
        if v is None:
            body += _pack_int32(-1)
        else:
            encoded = v.encode("utf-8")
            body += _pack_int32(len(encoded)) + encoded
    return b"D" + _pack_int32(4 + len(body)) + body


def _build_parameter_status(name: str, value: str) -> bytes:
    body = _encode_str(name) + _encode_str(value)
    return b"S" + _pack_int32(4 + len(body)) + body


# ---------------------------------------------------------------------------
# SQL parsing
# ---------------------------------------------------------------------------

_SELECT_KNN = re.compile(
    r"SELECT\s+\*\s+FROM\s+vectors\s+WHERE\s+collection\s*=\s*'([^']+)'"
    r"\s+ORDER\s+BY\s+embedding\s*<->\s*'(\[.*?\])'\s+LIMIT\s+(\d+)",
    re.IGNORECASE | re.DOTALL,
)

_SELECT_COUNT = re.compile(
    r"SELECT\s+count\s*\(\s*\*\s*\)\s+FROM\s+vectors\s+WHERE\s+collection\s*=\s*'([^']+)'",
    re.IGNORECASE,
)

_INSERT_VECTOR = re.compile(
    r"INSERT\s+INTO\s+vectors\s*\(\s*collection\s*,\s*embedding\s*,\s*metadata\s*\)"
    r"\s+VALUES\s*\(\s*'([^']+)'\s*,\s*'(\[.*?\])'\s*,\s*'(.*)'\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_vector(raw: str) -> List[float]:
    raw = raw.strip()
    if raw.startswith("["):
        raw = raw[1:]
    if raw.endswith("]"):
        raw = raw[:-1]
    return [float(x) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Client handler
# ---------------------------------------------------------------------------

class _ClientHandler:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        vector_service,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.vector_service = vector_service

    async def run(self) -> None:
        try:
            await self._handle_startup()
            await self._send(
                _build_parameter_status("server_version", "14.0")
                + _build_parameter_status("client_encoding", "UTF8")
                + _build_auth_ok()
                + _build_ready_for_query()
            )
            await self._query_loop()
        except asyncio.IncompleteReadError:
            pass
        except Exception:
            logger.exception("PgWire client error")
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass

    async def _send(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def _handle_startup(self) -> None:
        # Read 4-byte length
        length_bytes = await self.reader.readexactly(4)
        length = struct.unpack("!I", length_bytes)[0]
        if length < 4:
            return
        payload = await self.reader.readexactly(length - 4)
        # protocol version is first 4 bytes; ignore it (startup message)

    async def _query_loop(self) -> None:
        while True:
            msg_type = await self.reader.readexactly(1)
            length_bytes = await self.reader.readexactly(4)
            length = struct.unpack("!I", length_bytes)[0]
            body = await self.reader.readexactly(length - 4) if length > 4 else b""

            if msg_type == b"Q":
                query = body.rstrip(b"\x00").decode("utf-8", errors="replace")
                await self._handle_query(query)
            elif msg_type == b"X":  # Terminate
                break
            else:
                # Unknown message: send error + ready
                await self._send(
                    _build_error_response(f"Unsupported message type: {msg_type!r}")
                    + _build_ready_for_query()
                )

    async def _handle_query(self, query: str) -> None:
        query = query.strip()
        try:
            response = await self._dispatch(query)
        except Exception as exc:
            response = _build_error_response(str(exc))
        await self._send(response + _build_ready_for_query())

    async def _dispatch(self, query: str) -> bytes:
        # KNN search
        m = _SELECT_KNN.search(query)
        if m:
            collection, vec_raw, k = m.group(1), m.group(2), int(m.group(3))
            query_vector = _parse_vector(vec_raw)
            result = self.vector_service.search_vectors(
                query_vector=query_vector,
                k=k,
                method="brute",
                collection_id=collection,
            )
            rows = result.get("results", [])
            columns = ["vector_id", "score", "metadata"]
            rd = _build_row_description(columns)
            data_rows = b""
            for r in rows:
                data_rows += _build_data_row([
                    str(r.get("vector_id", "")),
                    str(r.get("score", "")),
                    json.dumps(r.get("metadata", {})),
                ])
            return rd + data_rows + _build_command_complete(f"SELECT {len(rows)}")

        # Count
        m = _SELECT_COUNT.search(query)
        if m:
            collection = m.group(1)
            result = self.vector_service.get_all_vectors(limit=1, offset=0, collection_id=collection)
            count = result.get("total", 0)
            rd = _build_row_description(["count"])
            dr = _build_data_row([str(count)])
            return rd + dr + _build_command_complete("SELECT 1")

        # Insert
        m = _INSERT_VECTOR.search(query)
        if m:
            collection, vec_raw, meta_raw = m.group(1), m.group(2), m.group(3)
            embedding = _parse_vector(vec_raw)
            try:
                metadata = json.loads(meta_raw) if meta_raw.strip() else {}
            except json.JSONDecodeError:
                metadata = {"raw": meta_raw}
            result = self.vector_service.create_vector(
                vector_data=embedding,
                metadata=metadata,
                collection_id=collection,
            )
            return _build_command_complete("INSERT 0 1")

        # Unrecognized but non-empty — return empty result set
        if query.upper().startswith("SELECT"):
            return _build_row_description(["result"]) + _build_command_complete("SELECT 0")

        return _build_command_complete("OK")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class PgWireServer:
    """Minimal async PostgreSQL wire protocol server proxying to VectorService."""

    def __init__(self, vector_service=None) -> None:
        self.vector_service = vector_service
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self, host: str = "0.0.0.0", port: int = 5433) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, host, port
        )
        logger.info("PgWire server listening on %s:%d", host, port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("PgWire server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        handler = _ClientHandler(reader, writer, self.vector_service)
        await handler.run()
