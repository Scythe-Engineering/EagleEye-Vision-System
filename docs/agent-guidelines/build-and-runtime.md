# Build and Runtime

## Python environment

- Use `uv` for Python dependency management and environment sync.
- Run `uv sync` before backend execution and tests.
- Do not use `pip install` for project dependency setup.

## Backend runtime

- Start backend with:

```bash
python src/main_backend.py
```

- Backend serves on port `5001`.

## Frontend runtime and build

- WebUI source root is `src/webui/`.
- Build frontend assets with:

```bash
npm run build
```

- Built assets are emitted to `src/webui/static/`.
- Flask serves WebUI from `src/webui/static/`, so backend UI flows require a
  completed frontend build.

## Development loop

- Frontend dev server:

```bash
npm run dev
```

- Vite root is `src/webui/` (not repository root).
- Vite excludes `static/**` and `web_server.py` from file watch.
