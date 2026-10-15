import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_hits = defaultdict(deque)


def hit(key: str, limit: int, window_s: int) -> bool:
    """True if the action is allowed (and records it); False when `key` exceeded `limit` per `window_s`."""
    now = time.time()
    with _lock:
        q = _hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True
