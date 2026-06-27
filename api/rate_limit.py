"""Shared slowapi limiter instance.

Import this in routers to add per-endpoint rate limits.
The SlowAPIMiddleware in main.py reads limits from decorated routes via app.state.limiter.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
