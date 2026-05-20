# AGENTS.md

EagleEye Vision System is a configurable FRC vision pipeline and WebUI stack
for camera ingest, detection, processing, and robot-facing output.

## Essentials (always relevant)

- **Package manager (Python)**: use `uv` (`uv sync`), never `pip install`.
- **Non-standard build dependency**: build WebUI with `npm run build` before
  running backend UI flows, because Flask serves built assets from
  `src/webui/static/`.
- **Engineering decision rule**: when multiple valid implementations exist,
  propose the simplest solution that satisfies requirements, and call out
  trade-offs if a more complex approach is chosen.

## Progressive disclosure docs

- [`docs/agent-guidelines/build-and-runtime.md`](docs/agent-guidelines/build-and-runtime.md)
- [`docs/agent-guidelines/operations-architecture.md`](docs/agent-guidelines/operations-architecture.md)
- [`docs/agent-guidelines/webui-integration.md`](docs/agent-guidelines/webui-integration.md)
- [`docs/agent-guidelines/code-style.md`](docs/agent-guidelines/code-style.md)
- [`docs/agent-guidelines/decision-making.md`](docs/agent-guidelines/decision-making.md)

## Cursor Cloud specific instructions

- **Starting the backend**: Run `PYTHONPATH=/workspace uv run python src/main_backend.py` from the repo root. The `PYTHONPATH` is required because `src/` is not a Python package (no `__init__.py`) yet all imports use `from src.…`.
- **Backend port**: Flask + SocketIO serves on port **5001** (`http://localhost:5001/`).
- **Startup auto-heals**: The `StartupInstallChecker` in `src/startup/install_check.py` automatically runs `uv sync`, `npm install`, and `npm run build` if it detects missing dependencies or stale build artifacts. Rust modules are built next via `maturin develop`.
- **No cameras expected**: In headless/cloud environments, the backend logs "No cameras detected" and skips pipeline startup. The WebUI still loads and system status APIs work normally.
- **Tests**: `uv run pytest tests/ -ra` — 135+ tests, all pass in headless mode. Two YOLO tests are skipped by marker.
- **Linter**: `uv run mypy src/ --ignore-missing-imports --explicit-package-bases` — runs successfully; pre-existing type errors are expected.
- **WebUI build**: `npm run build` outputs to `src/webui/static/`. Vite dev server (`npm run dev`) is optional and runs on port 5173.
- **Rust modules**: Two PyO3 modules (`temporal_acceleration`, `pose_outlier_filter`) are compiled via `maturin develop`. They are rebuilt automatically on backend startup if the source hash changes.
