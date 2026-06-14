"""Tests for Change Data Capture connector (Phase 3)."""

import os
import threading
import time
from unittest.mock import MagicMock

import pytest

from services.cdc_connector import CDCConfig, CDCEvent, CDCConnector


# ---- CDCConfig tests -------------------------------------------------------

def test_cdc_config_defaults():
    cfg = CDCConfig(name="test", collection_id="coll1", kafka_topic="db.public.table")
    assert cfg.name == "test"
    assert cfg.collection_id == "coll1"
    assert cfg.kafka_topic == "db.public.table"
    assert cfg.batch_size == 100
    assert cfg.flush_interval_sec == 5.0
    assert cfg.id_column == "id"
    assert cfg.text_columns == ["text", "content", "description", "title"]


def test_cdc_config_custom():
    cfg = CDCConfig(
        name="custom",
        collection_id="my_coll",
        kafka_topic="topic",
        batch_size=10,
        flush_interval_sec=1.0,
        field_mapping={"title": "name", "desc": "description"},
        text_columns=["name", "body"],
        id_column="uuid",
    )
    assert cfg.batch_size == 10
    assert cfg.field_mapping == {"title": "name", "desc": "description"}
    assert cfg.text_columns == ["name", "body"]
    assert cfg.id_column == "uuid"


# ---- CDCEvent tests --------------------------------------------------------

def test_cdc_event_create():
    payload = {"op": "c", "after": {"id": 1, "text": "hello", "price": 10.0}, "source": {"ts_ms": 1000}}
    event = CDCEvent(payload)
    assert event.is_create is True
    assert event.is_update is False
    assert event.is_delete is False
    assert event.op == "c"


def test_cdc_event_delete():
    payload = {"op": "d", "before": {"id": 5, "text": "bye"}, "source": {"ts_ms": 2000}}
    event = CDCEvent(payload)
    assert event.is_delete is True
    assert event.is_create is False


def test_cdc_event_update():
    payload = {"op": "u", "before": {"id": 3}, "after": {"id": 3, "text": "updated"}}
    event = CDCEvent(payload)
    assert event.is_update is True


def test_cdc_event_get_id_from_after():
    cfg = CDCConfig(name="t", collection_id="c", kafka_topic="t")
    payload = {"op": "c", "after": {"id": 42, "text": "data"}}
    event = CDCEvent(payload)
    assert event.get_id(cfg) == "42"


def test_cdc_event_get_id_from_before_when_no_after():
    cfg = CDCConfig(name="t", collection_id="c", kafka_topic="t")
    payload = {"op": "d", "before": {"id": 99}}
    event = CDCEvent(payload)
    assert event.get_id(cfg) == "99"


def test_cdc_event_get_text():
    cfg = CDCConfig(name="t", collection_id="c", kafka_topic="t",
                    text_columns=["title", "body"])
    payload = {"op": "c", "after": {"id": 1, "title": "Hello", "body": "World"}}
    event = CDCEvent(payload)
    text = event.get_text(cfg)
    assert "Hello" in text
    assert "World" in text


def test_cdc_event_get_metadata():
    cfg = CDCConfig(name="t", collection_id="c", kafka_topic="t",
                    field_mapping={"source_field": "name", "value_field": "score"})
    payload = {"op": "c", "after": {"id": 1, "name": "test", "score": 95}}
    event = CDCEvent(payload)
    meta = event.get_metadata(cfg)
    assert meta["source_field"] == "test"
    assert meta["value_field"] == 95
    assert "_cdc_op" in meta
    assert "_cdc_source" in meta


def test_cdc_event_empty_after():
    payload = {"op": "r", "after": {}}
    event = CDCEvent(payload)
    assert event.is_create is True
    assert event.get_id(CDCConfig(name="t", collection_id="c", kafka_topic="t")) == ""


# ---- CDCConnector tests ----------------------------------------------------

def test_cdc_connector_init(tmp_path):
    cfg = CDCConfig(name="test_conn", collection_id="c1", kafka_topic="t1")
    conn = CDCConnector(cfg)
    assert conn.config.name == "test_conn"
    assert conn._running is False
    assert len(conn._buffer) == 0


def test_cdc_connector_set_callbacks():
    cfg = CDCConfig(name="t", collection_id="c", kafka_topic="t")
    conn = CDCConnector(cfg)
    embed_fn = lambda texts: [[0.1] * 4 for _ in texts]
    ingest_fn = lambda cid, vec, meta: None
    delete_fn = lambda cid, vid: None
    conn.set_embed_fn(embed_fn)
    conn.set_ingest_fn(ingest_fn)
    conn.set_delete_fn(delete_fn)
    assert conn._embed_fn is not None
    assert conn._ingest_fn is not None
    assert conn._delete_fn is not None


def test_cdc_connector_enqueue_and_process(tmp_path):
    cfg = CDCConfig(name="test", collection_id="c1", kafka_topic="t1",
                    batch_size=2, flush_interval_sec=0.5)
    conn = CDCConnector(cfg)

    ingested = []
    deleted = []

    def ingest_fn(cid, vec, meta):
        ingested.append({"cid": cid, "vec": vec, "meta": meta})

    def delete_fn(cid, vid):
        deleted.append({"cid": cid, "vid": vid})

    conn.set_embed_fn(lambda texts: [[0.1] * 4 for _ in texts])
    conn.set_ingest_fn(ingest_fn)
    conn.set_delete_fn(delete_fn)

    # Enqueue a create event
    payload = {"op": "c", "after": {"id": 1, "text": "hello world", "title": "greeting"}}
    conn.enqueue_event(CDCEvent(payload))

    # Enqueue a delete event
    del_payload = {"op": "d", "before": {"id": 2}}
    conn.enqueue_event(CDCEvent(del_payload))

    # Flush should process batch
    conn._flush_buffer()

    assert len(ingested) == 1
    assert ingested[0]["cid"] == "c1"
    assert ingested[0]["meta"]["vector_id"].startswith("cdc_test_")

    assert len(deleted) == 1
    assert deleted[0]["vid"].startswith("cdc_test_")


def test_cdc_connector_batch_flush_on_size(tmp_path):
    cfg = CDCConfig(name="test", collection_id="c1", kafka_topic="t1",
                    batch_size=3)  # flush every 3 events
    conn = CDCConnector(cfg)

    ingested = []
    conn.set_embed_fn(lambda texts: [[float(i)] * 4 for i in range(len(texts))])
    conn.set_ingest_fn(lambda cid, vec, meta: ingested.append({"cid": cid, "vec": vec}))
    conn.set_delete_fn(lambda cid, vid: None)

    # Enqueue 2 events (no flush yet)
    for i in range(2):
        conn.enqueue_event(CDCEvent({"op": "c", "after": {"id": i, "text": f"doc{i}"}}))
    assert len(ingested) == 0  # not flushed yet

    # 3rd event triggers batch flush
    conn.enqueue_event(CDCEvent({"op": "c", "after": {"id": 2, "text": "doc2"}}))
    assert len(ingested) == 3  # was flushed


def test_cdc_connector_start_stop(tmp_path):
    cfg = CDCConfig(name="test", collection_id="c1", kafka_topic="t1",
                    batch_size=100, flush_interval_sec=1.0)
    conn = CDCConnector(cfg)
    conn.set_embed_fn(lambda texts: [[0.1] * 4 for _ in texts])
    conn.set_ingest_fn(lambda cid, vec, meta: None)
    conn.set_delete_fn(lambda cid, vid: None)

    conn.start()
    assert conn._running is True
    assert conn._thread is not None
    assert conn._thread.is_alive()

    time.sleep(0.3)  # Let thread start
    conn.stop()
    assert conn._running is False


def test_cdc_connector_get_status():
    cfg = CDCConfig(name="status_test", collection_id="c1", kafka_topic="t1")
    conn = CDCConnector(cfg)
    status = conn.get_status()
    assert status["name"] == "status_test"
    assert status["running"] is False
    assert status["buffer_size"] == 0


def test_cdc_connector_enqueue_many_events(tmp_path):
    cfg = CDCConfig(name="test", collection_id="c1", kafka_topic="t1",
                    batch_size=10, text_columns=["content"])
    conn = CDCConnector(cfg)

    ingested = []
    conn.set_embed_fn(lambda texts: [[0.1] * 4 for _ in texts])
    conn.set_ingest_fn(lambda cid, vec, meta: ingested.append({"cid": cid}))

    for i in range(25):
        conn.enqueue_event(CDCEvent({
            "op": "c",
            "after": {"id": i, "content": f"document number {i}"}
        }))

    # Should have been flushed (first batch at 10, second at 20)
    time.sleep(0.2)
    assert len(ingested) >= 20


def test_cdc_connector_no_embed_fn_skips(tmp_path):
    """Events should be skipped if no embed function is set."""
    cfg = CDCConfig(name="test", collection_id="c1", kafka_topic="t1",
                    batch_size=2)
    conn = CDCConnector(cfg)
    ingested = []
    conn.set_ingest_fn(lambda cid, vec, meta: ingested.append({"cid": cid}))

    for i in range(5):
        conn.enqueue_event(CDCEvent({
            "op": "c",
            "after": {"id": i, "text": f"doc{i}"}
        }))

    assert len(ingested) == 0  # nothing was ingested without embed_fn
