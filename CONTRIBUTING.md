# Contributing to Vector DB

Thank you for your interest! Contributions of all sizes are welcome.

## Setup

1. **Clone** the repository:
   ```powershell
   git clone https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB.git
   cd BUILDING-MY-OWN-VECTOR-DB
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install -e sdk  # install SDK in editable mode
   ```

4. **Configure environment**:
   Copy `.env.example` to `.env` and set your `DATABASE_URL` and other settings.

5. **Start PostgreSQL** and initialize the schema:
   ```powershell
   python -c "from database.schema import Base, engine; Base.metadata.create_all(engine)"
   ```

6. **Verify** with:
   ```powershell
   pytest test/ -v
   ```

## Running Tests

```powershell
pytest test/ -v                    # all tests
pytest sdk/tests/ -v               # SDK tests only
pytest test/ -k "hnsw" -v          # HNSW-specific tests
pytest test/ --cov=. --cov-report=html  # with coverage
```

## Code Style

- **Linting + formatting**: [Ruff](https://docs.astral.sh/ruff/)
  ```powershell
  ruff check .    # find issues
  ruff format .   # auto-format
  ```
- **Type hints**: Required for all public function signatures.
- **Line length**: 100 characters max.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.

## Pull Request Process

1. Create a **feature branch** from `main`:
   ```powershell
   git checkout -b feat/your-feature-name
   ```

2. Make your changes and **add tests** for new functionality.

3. Ensure **all tests pass** and lint is clean:
   ```powershell
   pytest test/ -v && ruff check .
   ```

4. Update documentation if your change affects the API surface
   (`readme.md`, `sdk/README.md`, or inline docstrings).

5. Commit with a clear message (Conventional Commits preferred):
   ```
   feat: add time-series vector insert endpoint
   fix: correct HNSW ef_search overflow on small datasets
   docs: document gRPC server setup
   ```

6. Push and open a Pull Request against `main`. Include a description of
   what the change does and why.

## Project Structure

```
api/          FastAPI routes, middleware, gRPC
config/       Settings, database connection, logging
database/     SQLAlchemy models and DB wrappers
models/       Pydantic schemas and vector model
services/     Business logic layer
utils/        Algorithms (HNSW, IVF, PQ, BM25, distance)
sdk/          Python client SDK
test/         Integration and unit tests
docs/         Design docs and guides
helm/         Kubernetes Helm chart
```

## Getting Help

- Open an issue for bugs or feature requests
- Check existing `test/` files for usage examples
- See `docs/` for design documentation

## Licensing

By contributing, you agree that your contributions will be licensed under
the MIT License (see `LICENSE`).
