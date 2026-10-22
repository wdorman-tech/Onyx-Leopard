# Repository Guidelines

## Project Structure & Module Organization

Onyx Leopard is split into a FastAPI backend and a Next.js frontend. Backend source lives in `backend/src/`: the app and routes in `backend/src/main.py`, simulation code in `backend/src/simulation/` (engine, CEOs, OpenRouter client, pure math in `dynamics.py`). Backend tests live in `backend/tests/`; bundled `.oleo` company profiles in `backend/companies/`. Frontend code lives in `frontend/src/`, with the single App Router page in `frontend/src/app/`, UI in `frontend/src/components/`, the SSE hook in `frontend/src/hooks/`, the API client in `frontend/src/lib/`, and graph types in `frontend/src/types/`. Longer notes are in `documentation/`, `references/`, and `research/`.

## Build, Test, and Development Commands

- `python start.py`: prompts for and verifies `OPENROUTER_API_KEY`, installs dependencies, and launches backend and frontend locally.
- `cd backend && uv run pytest`: runs the backend test suite.
- `cd backend && uv run pytest -m "not slow"`: skips tests marked as long-running.
- `cd backend && uv run ruff check src tests`: runs Python lint checks.
- `cd frontend && pnpm dev`: starts the Next.js development server.
- `cd frontend && pnpm build`: creates a production frontend build.
- `cd frontend && pnpm lint`: runs ESLint.

## Coding Style & Naming Conventions

Python targets 3.12+, uses Pydantic 2 models, and is linted with Ruff. Use snake_case for Python files, functions, and variables; use PascalCase for classes and Pydantic models. TypeScript components use PascalCase filenames such as `SimSetup.tsx`; hooks use camelCase names prefixed with `use`, such as `useSimulation.ts`.

Before editing frontend code, read `frontend/AGENTS.md`; this project uses Next.js 16, whose conventions may differ from older versions.

## Testing Guidelines

Add or update tests in `backend/tests/` for backend behavior changes. Match the `test_*.py` naming pattern and keep reusable fixtures in `backend/tests/fixtures/`. Prefer deterministic tests for simulation logic, and mark long-running cases with `@pytest.mark.slow`. For frontend changes, run `pnpm lint`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style messages, for example `feat(frontend): ...`, `fix(ai): ...`, `refactor(backend): ...`, and `test(engine): ...`. Keep commits scoped and descriptive. Pull requests should summarize behavior changes, list validation commands, link related issues or plans, and include screenshots for visible UI changes. Note any `.env` or migration steps.

## Security & Configuration Tips

Do not commit secrets. Use `.env.example` as the template and keep local credentials in `backend/.env`. The app runs with heuristic CEOs without an OpenRouter key; LLM behavior is tested against a mocked transport (`tests/conftest.py::FakeRouter`), and `OPENROUTER_BASE_URL` can point the backend at a local stand-in.
