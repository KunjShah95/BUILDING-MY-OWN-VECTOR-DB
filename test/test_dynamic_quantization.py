"""Tests for the dynamic quantization policy."""

import numpy as np

from utils.dynamic_quantization import QuantizationPolicy, Precision, _bytes_per_vector


def test_bytes_per_vector_tiers():
    assert _bytes_per_vector(Precision.FP32, 384) == 4 * 384
    assert _bytes_per_vector(Precision.INT8, 384) == 384
    assert _bytes_per_vector(Precision.BINARY, 384) == 48
    assert _bytes_per_vector(Precision.PQ, 384, pq_m=16) == 16


def test_recommend_no_pressure_keeps_fp32():
    qp = QuantizationPolicy()
    rec = qp.recommend(1000, 128, free_memory_fraction=0.9)
    assert rec["precision"] == "fp32"


def test_recommend_pressure_downgrades():
    qp = QuantizationPolicy()
    assert qp.recommend(1000, 128, free_memory_fraction=0.25)["precision"] == "int8"
    assert qp.recommend(1000, 128, free_memory_fraction=0.10)["precision"] == "pq"
    assert qp.recommend(1000, 128, free_memory_fraction=0.05)["precision"] == "binary"


def test_recommend_budget_selection():
    # 10k x 384-dim fp32 = ~14.6MB; a 1MB budget forces aggressive compression
    qp = QuantizationPolicy(memory_budget_mb=1.0)
    rec = qp.recommend(10000, 384, free_memory_fraction=0.9)
    assert rec["precision"] in ("pq", "binary")
    assert rec["compression_ratio"] > 1


def test_recommend_reports_compression():
    qp = QuantizationPolicy()
    rec = qp.recommend(1000, 128, free_memory_fraction=0.05)
    assert rec["fp32_mb"] > rec["estimated_mb"]
    assert rec["compression_ratio"] > 1


def test_apply_fp32_passthrough():
    qp = QuantizationPolicy()
    out = qp.apply([[1.0, 2.0, 3.0]], precision="fp32")
    assert out["precision"] == "fp32"
    assert out["data"].dtype == np.float32


def test_apply_int8_roundtrip_close():
    qp = QuantizationPolicy()
    vecs = [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]
    out = qp.apply(vecs, precision="int8")
    assert out["precision"] == "int8"
    assert out["data"].dtype == np.uint8
    # dequantize and check closeness
    decoded = out["data"].astype(np.float32) / 255.0 * out["scale"] + out["min"]
    assert np.allclose(decoded, np.array(vecs), atol=0.05)


def test_apply_binary_packs_sign_bits():
    qp = QuantizationPolicy()
    out = qp.apply([[1.0, -1.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0]], precision="binary")
    assert out["precision"] == "binary"
    assert out["dim"] == 8
    assert out["data"].shape == (1, 1)  # 8 bits -> 1 byte


def test_apply_pq_routes_out():
    qp = QuantizationPolicy()
    out = qp.apply([[1.0, 2.0]], precision="pq")
    assert out["route_to"] == "PQIndex"
