# WebUI Integration

## Build output contract

- Frontend build output directory is `src/webui/static/`.
- Flask serves WebUI assets from this directory.
- If assets are not built, pipeline editor UI will not be available.

## Path and root assumptions

- Vite project root is `src/webui/`.
- CSS/JS relative paths are resolved from `src/webui/`, not repository root.

## UI composition

- Handlebars partials are in:
  - `src/webui/html/tabs/`
  - `src/webui/html/partials/`

## Runtime interfaces

- Backend API base: `http://localhost:5001/`.
- Real-time pose updates use SocketIO.
- Three.js aliases are configured in `vite.config.js`.
