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
from dotenv import dotenv_values

log = logging.getLogger(__name__)

DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
APP_TITLE: Final[str] = "Onyx Leopard"
MODELS_PATH: Final[Path] = Path(__file__).resolve().parent / "models.yaml"
ENV_FILE: Final[Path] = Path(__file__).resolve().parents[2] / ".env"

REQUEST_TIMEOUT_S: Final[float] = 120.0
#: Attempts per call for transient failures (429, 5xx, timeouts).
MAX_ATTEMPTS: Final[int] = 3
RETRY_BACKOFF_S: Final[float] = 2.0
#: Output ceiling per call. Reasoning tokens count against it and some
#: providers ignore the requested effort (seen: GLM-5.3 Flash reasoning past
#: 4k tokens), so it is generous; billing is on tokens actually produced.
MAX_OUTPUT_TOKENS: Final[int] = 12_000
#: Extra attempts when reasoning ate the whole budget (empty content,
#: finish_reason="length"). A retry usually routes to another provider.
LENGTH_RETRIES: Final[int] = 1
CATALOG_TTL_S: Final[float] = 3600.0
#: A post-mortem prompt carries the decision log, so it costs about this many
#: average decision calls (measured: ~2-3x on GLM-5.3 Flash and MiMo-V2.6-Pro).
REPORT_COST_MULTIPLE: Final[float] = 3.0
#: Never hold back more than this share of the budget for reports.
MAX_REPORT_RESERVE_SHARE: Final[float] = 0.5
#: HTTP statuses that mean retrying this run is pointless (bad key, no credit).
FATAL_STATUSES: Final[frozenset[int]] = frozenset({401, 402, 403})
#: Reasoning efforts from cheapest to most expensive.
EFFORT_ORDER: Final[tuple[str, ...]] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class LlmError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def base_url() -> str:
    """OpenRouter endpoint; `OPENROUTER_BASE_URL` overrides it (proxies, local fakes)."""
    return os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def api_key() -> str | None:
    """Key from the environment, else read live from backend/.env (no restart needed)."""
    key = os.environ.get(API_KEY_ENV) or dotenv_values(ENV_FILE).get(API_KEY_ENV) or ""
    return key.strip() or None


#: OpenRouter keys are `sk-or-v1-` + hex; accept any URL-safe token so a
#: format change upstream doesn't lock users out, but nothing that could
#: break the .env line.
KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_\-]{20,200}")


def save_api_key(key: str) -> None:
    """Write `OPENROUTER_API_KEY=key` to backend/.env, keeping every other line."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if line.partition("=")[0].strip() != API_KEY_ENV]
    lines.append(f"{API_KEY_ENV}={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class KeyCheck:
    status: str  # "ok" | "rejected" | "unverified"
    credit_remaining: float | None = None


async def verify_key(key: str, http: httpx.AsyncClient | None = None) -> KeyCheck:
    """Ask OpenRouter about `key` without spending anything."""
    client = http or httpx.AsyncClient()
    try:
        resp = await client.get(
            f"{base_url()}/key", headers={"Authorization": f"Bearer {key}"}, timeout=15.0
        )
    except httpx.HTTPError as exc:
        log.warning("could not verify OpenRouter key: %r", exc)
        return KeyCheck("unverified")
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code in FATAL_STATUSES:
        return KeyCheck("rejected")
    if resp.status_code != 200:
        return KeyCheck("unverified")
    remaining = (_safe_json(resp).get("data") or {}).get("limit_remaining")
    return KeyCheck("ok", float(remaining) if isinstance(remaining, int | float) else None)


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
        resp = await client.get(f"{base_url()}/models", timeout=20.0)
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
    """Dollar ceiling shared by every LLM call in one simulation run.

    The estimated cost of the end-of-run CEO reports is held back from the
    decision calls, so post-mortems stay AI-written even when the decisions
    would otherwise spend the whole budget. Decision spend is paced linearly
    over the run (`on_pace`), so a budget too small for a monthly cadence
    stretches the cadence instead of going silent partway through.
    """

    ceiling_usd: float
    #: End-of-run reports to reserve for (one per company).
    report_calls: int = 0
    spent_usd: float = 0.0
    calls: int = 0

    @property
    def reserve_usd(self) -> float:
        if not self.report_calls or not self.calls:
            return 0.0
        per_report = self.spent_usd / self.calls * REPORT_COST_MULTIPLE
        return min(self.report_calls * per_report, self.ceiling_usd * MAX_REPORT_RESERVE_SHARE)

    @property
    def exhausted(self) -> bool:
        """No more decision calls; whatever is left is kept for the reports."""
        return self.spent_usd >= self.ceiling_usd - self.reserve_usd

    @property
    def spent(self) -> bool:
        """The hard ceiling: not even reports may call past this."""
        return self.spent_usd >= self.ceiling_usd

    def on_pace(self, day: int, max_days: int) -> bool:
        """Decision spend so far is within a straight-line share of the run."""
        spendable = self.ceiling_usd - self.reserve_usd
        return self.spent_usd <= spendable * min(1.0, day / max(1, max_days))

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

    @property
    def can_report(self) -> bool:
        """End-of-run reports may use the reserve the decisions left behind."""
        return self.disabled_reason is None and not self.budget.spent

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        use_reserve: bool = False,
    ) -> dict[str, Any]:
        if self.disabled_reason is not None:
            raise LlmError(self.disabled_reason, fatal=True)
        if self.budget.spent if use_reserve else self.budget.exhausted:
            raise LlmError(f"budget of ${self.budget.ceiling_usd:.2f} exhausted")

        for attempt in range(LENGTH_RETRIES + 1):
            data = await self._post_with_fallback(system, user, schema_name, schema)
            self._record_cost(data)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            if content:
                return parse_json_object(content)
            finish = choice.get("finish_reason")
            out_of_budget = self.budget.spent if use_reserve else self.budget.exhausted
            if finish != "length" or attempt == LENGTH_RETRIES or out_of_budget:
                raise LlmError(f"empty completion (finish_reason={finish})")
            log.info("%s spent its output on reasoning; retrying once", self.model.id)
        raise AssertionError("unreachable")

    async def _post_with_fallback(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return await self._post(self._body(system, user, schema_name, schema))
        except LlmError as exc:
            if not (self._structured and _unsupported_structured(str(exc))):
                raise
            log.info("%s has no structured-output provider; using prompt JSON", self.model.id)
            self._structured = False
            return await self._post(self._body(system, user, schema_name, schema))

    def _record_cost(self, data: dict[str, Any]) -> None:
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is None:
            cost = (
                usage.get("prompt_tokens", 0) * self.model.prompt_price
                + usage.get("completion_tokens", 0) * self.model.completion_price
            )
        self.budget.record(float(cost))

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
                    f"{base_url()}/chat/completions", json=body, headers=headers
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
