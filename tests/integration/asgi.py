# A minimal synchronous driver for the ASGI application under test.
#
# The requests below go through the real application — the same transport the
# HTTP server uses — rather than calling the graph directly, because the
# adapter is where caller authentication, the size cap and the structured
# parameter screen live, and a test that skipped it would prove nothing about
# what a caller actually receives.
#
# The framework's own test client wraps this same transport but routes it
# through a compatibility shim that emits a deprecation warning on the pinned
# dependency set; driving the transport directly keeps the suite warning-free
# and adds no dependency, since httpx is already required by the framework.
#
# The application declares no lifespan handlers, so nothing is skipped by
# calling the transport without a lifespan cycle.

import asyncio
from typing import Any, Dict, Mapping, Optional

import httpx


class Response:
    """The parts of a response the assertions in this suite need."""

    def __init__(self, raw: httpx.Response) -> None:
        self.status_code = raw.status_code
        self.text = raw.text
        self._raw = raw

    def json(self) -> Any:
        return self._raw.json()


class Client:
    """Synchronous facade over an ASGI application."""

    def __init__(self, app: Any) -> None:
        self._app = app

    def _request(
        self, method: str, path: str, json_body: Optional[Dict[str, Any]], headers: Mapping[str, str]
    ) -> Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
                return await client.request(method, path, json=json_body, headers=dict(headers))

        return Response(asyncio.run(_send()))

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Response:
        return self._request("POST", path, json, headers or {})

    def get(self, path: str, headers: Optional[Mapping[str, str]] = None) -> Response:
        return self._request("GET", path, None, headers or {})
