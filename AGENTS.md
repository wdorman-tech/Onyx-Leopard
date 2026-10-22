# Repository Guidelines

## Project Structure & Module Organization

Onyx Leopard is split into a FastAPI backend and a Next.js frontend. Backend source lives in `backend/src/`, with routes in `backend/src/routes/`, simulation code in `backend/src/simulation/`, and BioSim math/agent layers in `backend/src/biosim/`. Backend tests live in `backend/tests/`, including `backend/tests/biosim/`. Frontend code lives in `frontend/src/`, with App Router pages in `frontend/src/app/`, reusable UI in `frontend/src/components/`, hooks in `frontend/src/hooks/`, API clients in `frontend/src/lib/`, and shared types in `frontend/src/types/`. Static assets and preloaded `.oleo` profiles are under `frontend/public/`. Longer notes are in `documentation/`, `references/`, `research/`, and `plans/`.

## Build, Test, and Development Commands

- `python start.py`: installs dependencies as needed, prompts for configuration, and launches backend and frontend locally.
- `cd backend && uv run pytest`: runs the backend test suite.
- `cd backend && uv run pytest -m "not slow"`: skips tests marked as long-running.
- `cd backend && uv run ruff check .`: runs Python lint checks.
- `cd frontend && pnpm dev`: starts the Next.js development server.
- `cd frontend && pnpm build`: creates a production frontend build.
- `cd frontend && pnpm lint`: runs ESLint.

## Coding Style & Naming Conventions

Python targets 3.12+, uses Pydantic 2 models, and is linted with Ruff. Use snake_case for Python files, functions, and variables; use PascalCase for classes and Pydantic models. TypeScript components use PascalCase filenames such as `MarketDashboard.tsx`; hooks use camelCase names prefixed with `use`, such as `useUnifiedSimulation.ts`.

Before editing frontend code, read `frontend/AGENTS.md`; this project uses Next.js 16, whose conventions may differ from older versions.

## Testing Guidelines

Add or update tests in `backend/tests/` for backend behavior changes. Match the `test_*.py` naming pattern and keep reusable fixtures in `backend/tests/fixtures/`. Prefer deterministic tests for simulation logic, and mark long-running cases with `@pytest.mark.slow`. For frontend changes, run `pnpm lint`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style messages, for example `feat(math): ...`, `fix(math): ...`, `test(biosim): ...`, and `chore(biosim): ...`. Keep commits scoped and descriptive. Pull requests should summarize behavior changes, list validation commands, link related issues or plans, and include screenshots for visible UI changes. Note any `.env` or migration steps.

## Security & Configuration Tips

Do not commit secrets. Use `.env.example` as the template and keep local credentials in `backend/.env`. The app can run in pure-math mode without an Anthropic key; document when a change requires LLM-backed behavior to verify.
