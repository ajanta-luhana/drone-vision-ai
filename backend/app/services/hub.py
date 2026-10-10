"""Tiny WebSocket fan-out hub. Pipeline threads call publish(); the event loop delivers."""
import asyncio


class Hub:
    def __init__(self):
        self.loop = None
        self.clients: set = set()

    @property
    def has_clients(self) -> bool:
        return bool(self.clients)

    def bind(self, loop):
        self.loop = loop

    async def _send_all(self, msg: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    def publish(self, msg: dict):
        """Thread-safe; silently does nothing when nobody is listening."""
        if self.loop and self.clients and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_all(msg), self.loop)


hub = Hub()
