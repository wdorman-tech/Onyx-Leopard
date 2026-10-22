from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from src.simulation.engine import Company
from src.simulation.library import NodeLibrary, get_library
from src.simulation.llm import Budget, ModelInfo, OpenRouterClient
from src.simulation.seed import sample_seed
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import sample_stance

GLM_FLASH = ModelInfo(
    id="z-ai/glm-5.3-flash",
    name="Z.ai: GLM 5.3 Flash",
    prompt_price=0.03e-6,
    completion_price=0.083e-6,
    supported_parameters=frozenset({"reasoning", "response_format", "structured_outputs"}),
    reasoning_efforts=("max", "high", "low"),
)


@pytest.fixture
def library() -> NodeLibrary:
    return get_library()


def make_company(
    library: NodeLibrary,
    *,
    archetype: str = "bootstrap",
    company_archetype: str = "small_team",
    economics: str | None = "physical",
    seed_value: int = 1,
    company_id: str = "co-0",
) -> Company:
    rng = random.Random(seed_value)
    seed = sample_seed(company_archetype, rng, library, economics)  # type: ignore[arg-type]
    return Company(
        company_id=company_id,
        name=f"Test {company_id}",
        seed=seed,
        stance=sample_stance(archetype, rng),
        library=library,
        rng=random.Random(seed_value),
        shock_scheduler=ShockScheduler("company", seed_value, rate_multiplier=0.0),
    )


def completion(content: dict[str, Any] | str, cost: float = 0.0001) -> dict[str, Any]:
    """An OpenRouter chat-completion response body."""
    text = content if isinstance(content, str) else json.dumps(content)
    return {
        "id": "gen-test",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 200, "cost": cost},
    }


class FakeRouter:
    """httpx transport that records requests and replays queued responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.queue: list[httpx.Response] = []
        #: Answers requests once the queue is empty: request body → response JSON.
        self.respond: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def push(self, status: int = 200, body: dict[str, Any] | None = None) -> None:
        self.queue.append(httpx.Response(status, json=body or {}))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.queue:
            return self.queue.pop(0)
        if self.respond is None:
            raise AssertionError("FakeRouter got an unexpected request")
        return httpx.Response(200, json=self.respond(json.loads(request.content)))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter()


def make_llm(router: FakeRouter, *, budget: float = 1.0, model: ModelInfo = GLM_FLASH):
    return OpenRouterClient(
        api_key="sk-or-test", model=model, budget=Budget(ceiling_usd=budget), http=router.client()
    )


@pytest.fixture(autouse=True)
def isolated_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Keep the developer's real backend/.env key out of tests."""
    from src.simulation import llm

    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / ".env")


@pytest.fixture(autouse=True)
def isolated_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saved runs go to a temp dir, never backend/runs."""
    from src.simulation import export

    monkeypatch.setattr(export, "RUNS_DIR", tmp_path / "runs")
