from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

console = Console()

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    # truncated body should be retried
    return isinstance(exc, httpx.TransportError | ValueError)


def _render(template: str, sample: object) -> str:
    return template.format(**sample) if isinstance(sample, dict) else template.format(sample)


def _report(failed: int, total: int) -> None:
    if failed:
        console.print(f"[red]{failed:,} of {total:,} failed[/red]")


def _progress(enabled: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not enabled,
    )


class Inference:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:30000/v1",
        api_key: str | None = None,
        concurrency: int = 256,
        timeout: float = 180.0,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout
        self.retries = retries

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            limits=httpx.Limits(
                max_connections=self.concurrency, max_keepalive_connections=self.concurrency, keepalive_expiry=300.0
            ),
            # pool=None: queue rather than raise PoolTimeout under concurrency
            timeout=httpx.Timeout(self.timeout, connect=10.0, pool=None),
        )

    async def _post(self, client: httpx.AsyncClient, path: str, payload: dict) -> dict:
        async def once() -> dict:
            r = await client.post(path, json=payload)
            r.raise_for_status()
            return r.json()

        return await AsyncRetrying(
            retry=retry_if_exception(_transient),
            stop=stop_after_attempt(self.retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=30.0),
            reraise=True,
        )(once)

    async def _map(
        self,
        units: Sequence,
        handler: Callable[[httpx.AsyncClient, Any], Awaitable[dict]],
        description: str,
        total: int,
        show: bool,
    ) -> dict:
        """Run ``handler`` over ``units`` with a fixed pool of workers that all pull from a shared iterator"""
        results: dict[str, Any] = {}
        if not units:
            return results
        pending: Iterator = iter(units)

        with _progress(show) as progress:
            bar = progress.add_task(description, total=total)

            async def worker(client: httpx.AsyncClient) -> None:
                # no lock needed: next() on the iterator never awaits
                for unit in pending:
                    pairs = await handler(client, unit)
                    results.update(pairs)
                    progress.advance(bar, len(pairs))

            async with self._client() as client:
                n = min(self.concurrency, len(units))
                await asyncio.gather(*(worker(client) for _ in range(n)))

        return results

    async def generate(
        self,
        samples: Sequence,
        template: str = "{}",
        *,
        system: str | None = None,
        usage: bool = False,
        progress: bool = True,
        on_result: Callable[[str, Any], None] | None = None,
        **body: object,
    ) -> list:
        """One completion per sample, ``on_result(prompt, result)`` as each returns. Prompts are deduplicated"""
        prompts = [_render(template, s) for s in samples]
        unique = list(dict.fromkeys(prompts))  # identical prompts cost one call

        async def handler(client: httpx.AsyncClient, prompt: str) -> dict:
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            try:
                data = await self._post(
                    client, "/chat/completions", {"model": self.model, "messages": messages, **body}
                )
                content = data["choices"][0]["message"]["content"]
                if not usage:
                    return {prompt: content}
                counts = data.get("usage") or {}
                return {
                    prompt: {
                        "content": content,
                        "prompt_tokens": counts.get("prompt_tokens"),
                        "completion_tokens": counts.get("completion_tokens"),
                        "error": None,
                    }
                }
            except Exception as exc:
                if not usage:
                    return {prompt: None}
                return {
                    prompt: {"content": None, "prompt_tokens": None, "completion_tokens": None, "error": repr(exc)}
                }

        async def handle(client: httpx.AsyncClient, prompt: str) -> dict:
            pair = await handler(client, prompt)
            if on_result:
                on_result(prompt, pair[prompt])
            return pair

        done = await self._map(unique, handle, "Generating", len(unique), progress)
        failed = sum(1 for v in done.values() if (v["error"] if usage else v is None))
        _report(failed, len(unique))
        return [done[p] for p in prompts]

    async def embed(
        self, samples: Sequence, template: str = "{}", *, batch_size: int = 64, progress: bool = True, **body: object
    ) -> list[list[float] | None]:
        """One vector per sample, ``batch_size`` texts per request"""
        texts = [_render(template, s) for s in samples]
        unique = list(dict.fromkeys(texts))
        # group by length so each batch pads to similar width
        by_length = sorted(unique, key=len)
        batches = [by_length[i : i + batch_size] for i in range(0, len(by_length), batch_size)]

        async def handler(client: httpx.AsyncClient, batch: list[str]) -> dict:
            try:
                data = await self._post(client, "/embeddings", {"model": self.model, "input": batch, **body})
                items = sorted(data["data"], key=lambda d: d["index"])
                # a short reply is truncated: drop the whole batch
                if len(items) == len(batch):
                    return {t: i["embedding"] for t, i in zip(batch, items, strict=True)}
            except Exception:
                return dict.fromkeys(batch)
            return dict.fromkeys(batch)

        done = await self._map(batches, handler, "Embedding", len(unique), progress)
        _report(sum(v is None for v in done.values()), len(unique))
        return [done[t] for t in texts]
