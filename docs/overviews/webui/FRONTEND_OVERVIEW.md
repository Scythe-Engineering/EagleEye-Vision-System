# Frontend workflow

The browser entry point is `src/webui/js/main.js`. It initializes the sidebar, camera feeds, 3D view, pipeline editor, settings, connection status, and the `/sse/stream` connection.

## Change the interface

1. Edit a tab in `src/webui/html/tabs/` or a shared fragment in `html/partials/`.
2. Put behavior in the matching module under `js/`. Pipeline code belongs under `js/pipeline/`; settings, feeds, dropdowns, and shared UI each have their own directory.
3. Use `style.css` or the component sheets under `css/`.
4. Use the existing request helpers and `EventSource` handlers. HTTP carries commands and configuration. SSE carries live updates such as heartbeat, poses, detections, logs, profiling, system status, and pipeline errors.
5. Run the repository WebUI build before testing the Flask-served page. Flask maps the generated bundle to `/js/main.js` and generated CSS to `/style.css`.

The 3D view lives in `init3DView.js`. Field and robot selectors load GLB files through `/assets/...` and `/get-robot-file/...`. Coordinate conversion helpers are in `js/utils/fieldSpaceTransforms.js`; keep pose conversion there rather than duplicating it in event handlers.

The pipeline editor is split between `PipelineStore`, creator controllers, the flowchart renderer, nodes, connections, and the minimap. Change state through the store and creator actions so persistence, history, and rendering remain in sync.
