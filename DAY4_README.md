# DAY 4 — Building My Own Vector DB

Date: Day 4

## Goal
Continue implementing and testing the IVF (inverted file) index and clustering components for the vector database. Produce clear notes about what was added, how to run the code and tests, and what to tackle next.

## Highlights / Summary
- Implemented and improved IVF-related codepaths and integration between index and storage layers.
- Added/updated tests for IVF and clustering to validate behavior and edge cases.
- Small refactors to vector model and utilities for clearer separation of concerns.

## Files of interest
- `utils/ivf_index.py` — core IVF index implementation and helpers.
- `utils/clustering.py` — clustering utilities used by IVF (k-means / centroid computations).
- `database/ivf_database.py` — persistence/DB layer for IVF index metadata and vectors.
- `models/vector_model.py` — vector generation/normalization utilities.
- `test/test_ivf.py` — unit tests validating IVF behavior.
- `test/test_clustering.py` — clustering-related tests.
- `main.py` and `api/main.py` — example runners for local experimentation and API.

## How to run
1. Create and activate the virtualenv (if not already):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run tests:

```powershell
pytest -q
```

3. Run the example script:

```powershell
python main.py
```

4. Start the API (if needed):

```powershell
python api\main.py
```

## Notes / Known issues
- Tests cover core IVF behaviors but more coverage is needed for large-scale data and persistence recovery paths.
- Performance tuning (batching, memory usage) remains to be profiled.
- Consider more deterministic clustering initialization for reproducible tests.

## Next steps (Day 5)
- Add benchmarks for index build and query latency.
- Improve persistence and checkpointing of IVF metadata and vector shards.
- Add end-to-end example with synthetic dataset and visualization.

---

Short log: Created `DAY4_README.md` documenting changes, run instructions, and next steps.
