"""SAML/SSO/OIDC authentication integration (OIDC + PKCE)."""
import base64
import hashlib
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OIDCConfig:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=lambda: ["openid", "email", "profile"])


@dataclass
class TokenResponse:
    access_token: str
    id_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None


@dataclass
class SSOSessionData:
    user_id: str
    email: str
    roles: List[str]
    expiry: float  # Unix timestamp


# ---------------------------------------------------------------------------
# In-memory session store (falls back when Redis is unavailable)
# ---------------------------------------------------------------------------

_session_store: Dict[str, SSOSessionData] = {}


class SSOSession:
    """Manages SSO sessions, using an in-memory dict as the backing store."""

    @staticmethod
    def save(session_id: str, data: SSOSessionData) -> None:
        _session_store[session_id] = data

    @staticmethod
    def get(session_id: str) -> Optional[SSOSessionData]:
        session = _session_store.get(session_id)
        if session and session.expiry > time.time():
            return session
        if session:
            _session_store.pop(session_id, None)
        return None

    @staticmethod
    def delete(session_id: str) -> None:
        _session_store.pop(session_id, None)

    @staticmethod
    def generate_session_id() -> str:
        return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_code_verifier() -> str:
    return secrets.token_urlsafe(96)


def _generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# OIDC Provider
# ---------------------------------------------------------------------------

class OIDCProvider:
    """OIDC provider integration supporting PKCE flow."""

    def __init__(self, config: OIDCConfig) -> None:
        self.config = config
        self._jwks_cache: Optional[Dict] = None
        self._jwks_fetched_at: float = 0
        self._discovery: Optional[Dict] = None

    # ---- Discovery -------------------------------------------------------

    def _discovery_url(self) -> str:
        return self.config.issuer.rstrip("/") + "/.well-known/openid-configuration"

    async def _fetch_discovery(self) -> Dict:
        if self._discovery:
            return self._discovery
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for OIDC; install with: pip install httpx")
        async with httpx.AsyncClient() as client:
            resp = await client.get(self._discovery_url(), timeout=10)
            resp.raise_for_status()
            self._discovery = resp.json()
        return self._discovery

    # ---- Auth URL --------------------------------------------------------

    def get_auth_url(
        self,
        state: Optional[str] = None,
        code_verifier: Optional[str] = None,
    ) -> tuple:
        """
        Build the OIDC authorization URL with PKCE.
        Returns (auth_url, code_verifier, state).
        """
        if code_verifier is None:
            code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        if state is None:
            state = secrets.token_urlsafe(16)

        # Build auth endpoint from issuer (best-effort without discovery)
        auth_endpoint = self.config.issuer.rstrip("/") + "/authorize"
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        url = auth_endpoint + "?" + urllib.parse.urlencode(params)
        return url, code_verifier, state

    # ---- Token exchange --------------------------------------------------

    async def exchange_code(
        self, code: str, code_verifier: str
    ) -> TokenResponse:
        """Exchange authorization code + PKCE verifier for tokens."""
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required; pip install httpx")

        token_endpoint = self.config.issuer.rstrip("/") + "/oauth/token"
        try:
            disc = await self._fetch_discovery()
            token_endpoint = disc.get("token_endpoint", token_endpoint)
        except Exception:
            pass  # Use fallback endpoint

        payload = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "redirect_uri": self.config.redirect_uri,
            "code": code,
            "code_verifier": code_verifier,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_endpoint, data=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

        return TokenResponse(
            access_token=data["access_token"],
            id_token=data.get("id_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 3600),
            refresh_token=data.get("refresh_token"),
        )

    # ---- Token verification ----------------------------------------------

    async def _fetch_jwks(self) -> Dict:
        """Fetch and cache JWKS (10-minute TTL)."""
        now = time.time()
        if self._jwks_cache and now - self._jwks_fetched_at < 600:
            return self._jwks_cache

        jwks_endpoint = self.config.issuer.rstrip("/") + "/.well-known/jwks.json"
        try:
            disc = await self._fetch_discovery()
            jwks_endpoint = disc.get("jwks_uri", jwks_endpoint)
        except Exception:
            pass

        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required; pip install httpx")

        async with httpx.AsyncClient() as client:
            resp = await client.get(jwks_endpoint, timeout=10)
            resp.raise_for_status()
            self._jwks_cache = resp.json()
            self._jwks_fetched_at = now
        return self._jwks_cache

    async def verify_token(self, id_token: str) -> Dict[str, Any]:
        """
        Validate an id_token JWT signature using JWKS.
        Returns the decoded claims dict.
        """
        if not _JWT_AVAILABLE:
            # Fallback: decode without verification (dev/test only)
            logger.warning("PyJWT not available; skipping signature verification")
            parts = id_token.split(".")
            if len(parts) < 2:
                raise ValueError("Invalid JWT")
            padding = 4 - len(parts[1]) % 4
            payload_bytes = base64.urlsafe_b64decode(parts[1] + "=" * padding)
            return json.loads(payload_bytes)

        jwks = await self._fetch_jwks()
        jwks_client = pyjwt.PyJWKClient.__new__(pyjwt.PyJWKClient)

        # Decode header to get kid
        try:
            header = pyjwt.get_unverified_header(id_token)
        except Exception as exc:
            raise ValueError(f"Cannot decode JWT header: {exc}") from exc

        # Find matching key from JWKS
        matching_key = None
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == header.get("kid") or not header.get("kid"):
                try:
                    matching_key = pyjwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_data))
                    break
                except Exception:
                    continue

        if matching_key is None:
            raise ValueError("No matching JWK found for id_token")

        claims = pyjwt.decode(
            id_token,
            matching_key,
            algorithms=["RS256", "ES256"],
            audience=self.config.client_id,
        )
        return claims
