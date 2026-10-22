"""OpenRouter client, model catalog, and per-run spend budget.

One OpenAI-compatible endpoint serves every model (Chinese open-weight or
Anthropic). Requests ask for strict JSON-schema output and are routed only
to providers that support it (`provider.require_parameters`). If a model has
no such provider, the client falls back to schema-in-prompt JSON for the rest
of the run. Cost comes from OpenRouter's `usage.cost`, so no local price
table can drift.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import yaml

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
APP_TITLE: Final[str] = "Onyx Leopard"
MODELS_PATH: Final[Path] = Path(__file__).resolve().parent / "models.yaml"

REQUEST_TIMEOUT_S: Final[float] = 120.0
#: Attempts per call for transient failures (429, 5xx, timeouts).
MAX_ATTEMPTS: Final[int] = 3
RETRY_BACKOFF_S: Final[float] = 2.0
#: Output ceiling per call. Reasoning tokens count against it, so it is
#: generous; billing is on tokens actually produced.
MAX_OUTPUT_TOKENS: Final[int] = 4096
CATALOG_TTL_S: Final[float] = 3600.0
#: HTTP statuses that mean retrying this run is pointless (bad key, no credit).
FATAL_STATUSES: Final[frozenset[int]] = frozenset({401, 402, 403})
#: Reasoning efforts from cheapest to most expensive.
EFFORT_ORDER: Final[tuple[str, ...]] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class LlmError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


# ─── Model catalog ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    prompt_price: float = 0.0  # USD per input token
    completion_price: float = 0.0  # USD per output token
    context_length: int = 0
    supported_parameters: frozenset[str] = frozenset()
    reasoning_efforts: tuple[str, ...] = ()
    intelligence_index: float | None = None
    created: int = 0  # unix seconds; newest models list first

    @classmethod
    def from_openrouter(cls, m: dict[str, Any]) -> ModelInfo:
        pricing = m.get("pricing") or {}
        reasoning = m.get("reasoning") or {}
        bench = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        return cls(
            id=m["id"],
            name=m.get("name") or m["id"],
            prompt_price=float(pricing.get("prompt") or 0.0),
            completion_price=float(pricing.get("completion") or 0.0),
            context_length=int(m.get("context_length") or 0),
            supported_parameters=frozenset(m.get("supported_parameters") or ()),
            reasoning_efforts=tuple(reasoning.get("supported_efforts") or ()),
            intelligence_index=bench.get("intelligence_index"),
            created=int(m.get("created") or 0),
        )

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt_per_million": round(self.prompt_price * 1e6, 4),
            "completion_per_million": round(self.completion_price * 1e6, 4),
            "context_length": self.context_length,
            "intelligence_index": self.intelligence_index,
        }


@dataclass
class Catalog:
    default_model: str
    reasoning_effort: str
    tiers: list[dict[str, Any]]
    anthropic: list[ModelInfo]
    models: dict[str, ModelInfo]
    live: bool

    def info(self, model_id: str) -> ModelInfo:
        return self.models.get(model_id) or ModelInfo(id=model_id, name=model_id)

    def public(self) -> dict[str, Any]:
        return {
            "default": self.default_model,
            "live": self.live,
            "tiers": [
                {**{k: t[k] for k in ("tier", "label", "blurb")}, **self.info(t["model"]).public()}
                for t in self.tiers
            ],
            "anthropic": [m.public() for m in self.anthropic],
        }


_catalog_cache: tuple[float, Catalog] | None = None


async def load_catalog(http: httpx.AsyncClient | None = None) -> Catalog:
    """Curated tiers + live OpenRouter metadata. Cached; degrades to static offline."""
    global _catalog_cache
    if _catalog_cache and time.monotonic() - _catalog_cache[0] < CATALOG_TTL_S:
        return _catalog_cache[1]

    config = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    prefix = config["anthropic_prefix"]
    models: dict[str, ModelInfo] = {}
    live = False
    client = http or httpx.AsyncClient()
    try:
        resp = await client.get(f"{OPENROUTER_BASE_URL}/models", timeout=20.0)
        resp.raise_for_status()
        for m in resp.json()["data"]:
            models[m["id"]] = ModelInfo.from_openrouter(m)
        live = True
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("OpenRouter model catalog unavailable (%s); using static list", exc)
    finally:
        if http is None:
            await client.aclose()

    if live:
        anthropic = sorted(
            (m for mid, m in models.items() if mid.startswith(prefix) and ":" not in mid),
            key=lambda m: (-m.created, m.id),
        )
    else:
        anthropic = [ModelInfo(id=mid, name=mid) for mid in config["anthropic_fallback"]]

    catalog = Catalog(
        default_model=config["default"],
        reasoning_effort=config["reasoning_effort"],
        tiers=config["tiers"],
        anthropic=anthropic,
        models=models,
        live=live,
    )
    if live:
        _catalog_cache = (time.monotonic(), catalog)
    return catalog


# ─── Spend budget ───────────────────────────────────────────────────────────


@dataclass
class Budget:
    """Dollar ceiling shared by every LLM call in one simulation run."""

    ceiling_usd: float
    spent_usd: float = 0.0
    calls: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.ceiling_usd

    def record(self, cost_usd: float) -> None:
        self.spent_usd += max(0.0, cost_usd)
        self.calls += 1


# ─── Client ─────────────────────────────────────────────────────────────────


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from model output (tolerates fences/preamble)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LlmError(f"no JSON object in model output: {text[:200]!r}")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LlmError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise LlmError("model output JSON is not an object")
    return obj


@dataclass
class OpenRouterClient:
    api_key: str
    model: ModelInfo
    budget: Budget
    reasoning_effort: str = "low"
    http: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    )
    #: Set after a fatal error (bad key, no credit); every later call is skipped.
    disabled_reason: str | None = None
    _structured: bool = True

    @property
    def available(self) -> bool:
        return self.disabled_reason is None and not self.budget.exhausted

    async def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        if self.disabled_reason is not None:
            raise LlmError(self.disabled_reason, fatal=True)
        if self.budget.exhausted:
            raise LlmError(f"budget of ${self.budget.ceiling_usd:.2f} exhausted")

        try:
            data = await self._post(self._body(system, user, schema_name, schema))
        except LlmError as exc:
            if not (self._structured and _unsupported_structured(str(exc))):
                raise
            log.info("%s has no structured-output provider; using prompt JSON", self.model.id)
            self._structured = False
            data = await self._post(self._body(system, user, schema_name, schema))

        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is None:
            cost = (
                usage.get("prompt_tokens", 0) * self.model.prompt_price
                + usage.get("completion_tokens", 0) * self.model.completion_price
            )
        self.budget.record(float(cost))

        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if not content:
            raise LlmError(f"empty completion (finish_reason={choice.get('finish_reason')})")
        return parse_json_object(content)

    def _body(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._structured:
            system += (
                "\n\nRespond with a single JSON object and nothing else. It must match "
                f"this JSON schema:\n{json.dumps(schema, separators=(',', ':'))}"
            )
        body: dict[str, Any] = {
            "model": self.model.id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if self._structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
            body["provider"] = {"require_parameters": True}
        reasoning = self._reasoning()
        if reasoning is not None:
            body["reasoning"] = reasoning
        return body

    def _reasoning(self) -> dict[str, Any] | None:
        if "reasoning" not in self.model.supported_parameters:
            return None
        effort = self.reasoning_effort
        efforts = self.model.reasoning_efforts
        if efforts and effort not in efforts:
            effort = next((e for e in EFFORT_ORDER if e in efforts and e != "none"), efforts[0])
        return {"effort": effort, "exclude": True}

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": APP_TITLE}
        last_error = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await self.http.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers
                )
            except httpx.TransportError as exc:
                last_error = f"network error: {exc!r}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:  # provider error delivered with HTTP 200
                        last_error = _error_message(data)
                    else:
                        return data
                elif resp.status_code in FATAL_STATUSES:
                    self.disabled_reason = (
                        f"OpenRouter rejected the request ({resp.status_code}): "
                        f"{_error_message(_safe_json(resp))}"
                    )
                    raise LlmError(self.disabled_reason, fatal=True)
                elif resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {_error_message(_safe_json(resp))}"
                else:
                    raise LlmError(f"HTTP {resp.status_code}: {_error_message(_safe_json(resp))}")
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
        raise LlmError(f"{self.model.id} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    async def aclose(self) -> None:
        await self.http.aclose()


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {"error": {"message": resp.text[:300]}}
    return data if isinstance(data, dict) else {}


def _error_message(data: dict[str, Any]) -> str:
    err = data.get("error") or {}
    if isinstance(err, dict):
        raw = (err.get("metadata") or {}).get("raw")
        return str(err.get("message") or "unknown error") + (f" ({raw})" if raw else "")
    return str(err)


def _unsupported_structured(message: str) -> bool:
    lowered = message.lower()
    return any(
        s in lowered for s in ("no endpoints found", "response_format", "json_schema", "structured")
    )
