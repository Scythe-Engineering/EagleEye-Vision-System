# WebUI

The WebUI runs on the Flask server at port 5001. It provides camera views, a Three.js field view, pipeline editing, profiling, calibration, asset management, network setup, logs, and system controls.

## Architecture

`src/webui/web_server.py` creates `EagleEyeInterface`, registers HTTP routes, and combines route handlers from `web_server_utils/`. The server supplies the built page and assets, MJPEG camera and visualization streams, JSON APIs, and a single-client server-sent event stream.

The browser starts in `js/main.js`. It uses plain JavaScript modules for the sidebar, camera feeds, 3D view, settings, and pipeline editor. `EventSource` receives live state from `/sse/stream`; ordinary changes use HTTP requests. Flask also initializes Socket.IO, but the bundled frontend uses SSE.

The production page loads `static/bundle.js` through `/js/main.js` and `static/main.css` through `/style.css`. Edit the source under `js/`, `html/`, and `css/`, then rebuild the static output.

## Guides

- [API reference](API_DOCUMENTATION.md)
- [Frontend workflow](FRONTEND_OVERVIEW.md)
- [Pipeline flowchart](PIPELINE_FLOWCHART_INTERFACE.md)
- [Assets](ASSETS_OVERVIEW.md)
