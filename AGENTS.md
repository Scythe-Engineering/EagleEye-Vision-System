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
