from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import time

from .contracts import GenerationRequest
from .service import ImageGenerationService, ServiceResponse


@dataclass
class _Pending:
    request: GenerationRequest
    future: asyncio.Future[ServiceResponse]


class AsyncBatchingGateway:
    """Collect concurrent HTTP requests into short windows before service planning."""

    def __init__(
        self,
        service: ImageGenerationService,
        *,
        max_delay_ms: int,
        max_queue_size: int,
        max_drain_size: int = 64,
    ) -> None:
        if max_delay_ms < 0 or max_queue_size <= 0 or max_drain_size <= 0:
            raise ValueError("gateway limits are invalid")
        self.service = service
        self.max_delay_ms = max_delay_ms
        self.max_drain_size = max_drain_size
        self.queue: asyncio.Queue[_Pending | None] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="image-microbatch-gateway")

    async def close(self) -> None:
        if self._worker is not None:
            await self.queue.put(None)
            await self._worker
            self._worker = None

    async def generate(self, request: GenerationRequest) -> ServiceResponse:
        if self._worker is None:
            raise RuntimeError("gateway has not been started")
        loop = asyncio.get_running_loop()
        now_ms = int(time.monotonic() * 1000)
        if request.submitted_at_ms == 0:
            request = replace(request, submitted_at_ms=now_ms)
        future: asyncio.Future[ServiceResponse] = loop.create_future()
        try:
            self.queue.put_nowait(_Pending(request, future))
        except asyncio.QueueFull as exc:
            raise RuntimeError("gateway_queue_full") from exc
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=request.deadline_ms / 1000.0)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("request_deadline_exceeded") from exc

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            if first is None:
                return
            pending = [first]
            cutoff = loop.time() + self.max_delay_ms / 1000.0
            while len(pending) < self.max_drain_size:
                remaining = cutoff - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self.queue.put(None)
                    break
                pending.append(item)
            now_ms = int(time.monotonic() * 1000)
            responses = self.service.submit_many((item.request for item in pending), now_ms=now_ms)
            by_id = {response.request_id: response for response in responses}
            for item in pending:
                if not item.future.cancelled():
                    item.future.set_result(by_id[item.request.request_id])

