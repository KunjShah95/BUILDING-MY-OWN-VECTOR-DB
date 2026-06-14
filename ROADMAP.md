# Vector DB Roadmap

**Legend:** ✅ done · 🟡 in progress · ⚪ planned

## Phase 10: Billion-Scale Performance ⚪
Target 100M–1B vectors with recall >95% and latency <10ms.
- ⚪ Vamana/DiskANN tuning on BIGANN, SPACEV, DEEP1B
- ⚪ PQ compression for 10-48x memory reduction at scale
- ⚪ Multi-level caching hierarchy (L1 RAM → L2 NVMe → L3 S3)
- ⚪ Adaptive batch sizing and connection pooling for 10K+ QPS

## Phase 11: Multi-Region Active-Active ⚪
Global scale with CRDT-based vector sync across regions.
- ⚪ CRDT merge logic for concurrent vector writes
- ⚪ Region-aware query routing (geo-proximity)
- ⚪ Conflict resolution with vector-clock timestamps
- ⚪ Cross-region replication monitoring dashboard

## Phase 12: Enterprise Hardening ⚪
Production isolation and compliance for regulated industries.
- ⚪ Tenant-level dedicated schemas + cgroup isolation
- ⚪ SQL-over-vector-db via pgvector wire-compatible layer
- ⚪ SAML/SSO/OIDC integration for auth
- ⚪ Audit dashboard with SOC2/GDPR export wizards

## Phase 13: Plugin SDK & Ecosystem ⚪
Hot-loadable custom index algorithms, encoders, and storage backends.
- ⚪ Plugin registry with versioned manifest
- ⚪ Plugin SDK for Python/Go/Rust
- ⚪ Marketplace for community plugins
- ⚪ Plugin sandbox (resource limits, crash isolation)

## Phase 14: Intelligent Query Mesh ⚪
Self-optimizing query routing and cost governance.
- ⚪ Query cost predictor (estimated scan count before execution)
- ⚪ Budget-aware query scheduler with tenant credit pools
- ⚪ Cross-index fusion with real-time latency/recall telemetry
- ⚪ Materialized view auto-recommendation engine

## Phase 15: Vector-Native AI Features ⚪
Differentiated AI capabilities beyond basic ANN search.
- ⚪ Fine-tuning interface for ranking models (LTR/ColBERT)
- ⚪ RLHF feedback loop from user click/skip signals
- ⚪ On-device embedding with federated update sync
- ⚪ Vector explanation (why this result?) via attention attribution

## Phase 16: Managed Cloud Platform ⚪
Self-service deployment, monitoring, and billing.
- ⚪ Helm chart auto-scaling with spot/preemptible nodes
- ⚪ Usage metering and per-tenant billing API
- ⚪ Web admin console with query analyzer and cost explorer
- ⚪ One-click restore from backup with point-in-time recovery
