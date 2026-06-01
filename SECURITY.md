# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 1.0.x   | ✅ Active support |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.

Instead, report it by:
1. Opening an issue with the label `security` (visible to maintainers only).
2. Emailing the maintainer directly (see GitHub profile for contact).

We aim to acknowledge reports within 48 hours and provide a fix timeline within
5 business days.

## Authentication

This project uses **API key authentication** via the `X-API-Key` header.

- Keys are stored as **SHA-256 hashes** — plaintext keys are never persisted.
- Key permissions are scoped per-collection (`read`, `write`, `read_write`).
- Use the `/api/keys/create` and `/api/keys/revoke` endpoints to manage keys.

## Production Deployment

- **Always use HTTPS** in production. The API does not encrypt traffic on its own.
- Set a **strong, unique `API_KEY`** via environment variable.
- Configure `ALLOWED_HOSTS` to restrict CORS origins.
- Set `DEBUG=false` in production to disable verbose error responses.
- Use the Docker Compose stack with network isolation between services.

## Dependency Management

- Keep dependencies up to date (`pip-audit` or Dependabot).
- Review new dependencies for known CVEs before adding them to `requirements.txt`.

## Data Protection

- Vector data is stored in PostgreSQL — configure encryption at rest if
  required by your compliance framework.
- Media files (uploaded images/audio) are stored on local disk or S3/Azure Blob;
  set appropriate bucket policies to restrict access.
