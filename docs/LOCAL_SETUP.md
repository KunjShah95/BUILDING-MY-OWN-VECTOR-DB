# Local setup (Windows / Linux)

Step-by-step guide to run PostgreSQL, the API, the SDK demo, and tests on your machine.

## Prerequisites

- Python 3.9+
- Docker Desktop (for Postgres via Compose) or a local PostgreSQL 12+ instance
- PowerShell (Windows) or bash (Linux/macOS)

## 1. PostgreSQL with Docker Compose

From the project root:

```powershell
cd "c:\BUILDING MY OWN VECTOR DB"
docker compose up -d postgres
```

Wait until healthy:

```powershell
docker compose ps
```

Default credentials (match `.env.example`):

| Setting  | Value            |
|----------|------------------|
| Host     | `localhost`      |
| Port     | `5432`           |
| Database | `vector_db`      |
| User     | `vector_user`    |
| Password | `vector_password`|

## 2. Environment variables

```powershell
copy .env.example .env
```

Edit `.env` if your Postgres URL differs:

```env
DATABASE_URL=postgresql://vector_user:vector_password@localhost:5432/vector_db
```

## 3. Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e sdk
```

**CPU-only PyTorch (recommended on Windows):** if `sentence-transformers` pulls a large CUDA torch build, install CPU wheels first:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 5. Create tables (migrate / init)

```powershell
python -c "from config.database import Base, engine; Base.metadata.create_all(bind=engine)"
```

## 6. Run the API

**Helper script (Windows):**

```powershell
.\scripts\run_local.ps1
```

**Helper script (Linux/macOS):**

```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

**Manual:**

```powershell
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Swagger: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## 7. Run the SDK demo

With the API running in another terminal:

```powershell
python examples/multimodal_demo.py
# or: python examples/multimodal_demo.py http://localhost:8000
```

The demo creates a text collection, ingests two documents, searches, builds a per-collection HNSW index, and searches again with `method=hnsw`.

## 8. Run tests

Multimodal integration tests need PostgreSQL. The test DB URL is in `test/test_multimodal.py` (`vector_db_test` by default). Create it if needed:

```sql
CREATE DATABASE vector_db_test;
```

Run:

```powershell
pytest test/test_multimodal.py test/test_multimodal_media.py -q
```

All unit tests (no live DB for some):

```powershell
pytest test/ -q
```

## Per-collection index (API)

```http
POST /collections/{collection_id}/index
GET  /collections/{collection_id}/index/stats
```

Index files are stored under `indexes/{collection_id}/hnsw_index_data.json`.

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `connection refused` on 5432 | Start Postgres: `docker compose up -d postgres` |
| Tests skipped (DB unavailable) | Create `vector_db_test` and fix credentials in test URL |
| Slow first ingest | First run downloads embedding models (~hundreds of MB) |
| Torch CUDA mismatch | Use CPU torch index URL above |
| Port 8000 in use | Change port: `--port 8001` and point SDK at that URL |
