"""Password hashing, JWT, RBAC permission map and login throttling."""
import hashlib
import hmac
import os
import time
from collections import defaultdict, deque

import jwt

from .config import settings

PERMISSIONS = {
    "admin": {"*"},
    "operator": {"missions:read", "missions:write", "videos:upload", "videos:process", "zones:read",
                 "events:read", "incidents:read", "incidents:review", "reports:read", "reports:create",
                 "assistant:use", "analytics:read", "system:read"},
    "analyst": {"missions:read", "zones:read", "events:read", "incidents:read", "reports:read",
                "reports:create", "assistant:use", "analytics:read", "system:read"},
    "viewer": {"missions:read", "zones:read", "events:read", "incidents:read", "reports:read", "analytics:read"},
}


def has_permission(role: str, perm: str) -> bool:
    p = PERMISSIONS.get(role, set())
    return "*" in p or perm in p


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, dk_hex = stored.split("$")
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


def create_token(username: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + settings.jwt_ttl_minutes * 60},
                      settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


class LoginThrottle:
    """Tiny in-memory brute-force protection: N failures per window per key."""

    def __init__(self, max_fail=5, window_s=300):
        self.max_fail, self.window_s = max_fail, window_s
        self.fails = defaultdict(deque)

    def _prune(self, key):
        q, now = self.fails[key], time.time()
        while q and now - q[0] > self.window_s:
            q.popleft()

    def blocked(self, key) -> bool:
        self._prune(key)
        return len(self.fails[key]) >= self.max_fail

    def fail(self, key):
        self.fails[key].append(time.time())

    def reset(self, key):
        self.fails.pop(key, None)


throttle = LoginThrottle()
