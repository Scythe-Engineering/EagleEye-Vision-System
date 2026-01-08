---
name: webui-development
description: Build and modify the WebUI frontend including Vite build configuration, Tailwind CSS, Three.js 3D visualization, and Handlebars templating
---

# WebUI Development Skill

Use this skill when working on the EagleEye WebUI frontend system.

## Critical Build Flow

**Development**:
```bash
npm install
npm run dev              # Vite dev server on port 5173
```

**Production Build**:
```bash
npm run build            # Outputs to src/webui/static/
npm run watch           # Continuous rebuild
```

⚠️ **Critical**: Flask backend serves WebUI from `src/webui/static/`. Must run `npm run build` before Flask can serve the pipeline editor UI.

## Vite Configuration

Vite root is `src/webui/` (NOT project root):
- All relative paths in config are relative to `src/webui/`
- Output: `src/webui/static/` (served by Flask)
- Excluded from watch: `static/**`, `web_server.py`
- Build tool: rolldown (not Webpack)

```javascript
// vite.config.js uses this root:
root: path.resolve(__dirname, "./src/webui")
// All paths resolve relative to src/webui/, not project root
```

## Tailwind CSS v4

Integrated via `@tailwindcss/vite` plugin (not PostCSS):
- Standard Tailwind classes work as expected
- No additional configuration needed
- tabWidth: 4 in package.json

## Handlebars Templating

Partials configured in two directories:
```javascript
partialDirectory: [
    "./src/webui/html/tabs",      // Tab/page templates
    "./src/webui/html/partials"   // Reusable components
]
```

**File locations**:
- Pages: `src/webui/html/tabs/*.html`
- Components: `src/webui/html/partials/*.html`
- Styles: `src/webui/css/*.css`
- JavaScript: `src/webui/js/**/*.js` (ES6 modules)

## Three.js Setup

Configured with aliases in `vite.config.js`:
```javascript
alias: {
    three: "node_modules/three/build/three.module.js",
    OrbitControls: "three/examples/jsm/controls/OrbitControls.js",
    GLTFLoader: "three/examples/jsm/loaders/GLTFLoader.js",
    DRACOLoader: "three/examples/jsm/loaders/DRACOLoader.js"
}
```

Import using alias names:
```javascript
import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { GLTFLoader } from 'GLTFLoader';
import { DRACOLoader } from 'DRACOLoader';
```

## Real-time Communication

Socket.IO client connects to Flask backend (port 5001):
```javascript
// Backend broadcasts pose updates
// Frontend listens on 'update_robot_transform' event
socket.on('update_robot_transform', (data) => {
    // Handle pose update
});
```

## Project Structure

```
src/webui/
├── index.html           # Main template
├── style.css            # Global styles
├── js/
│   ├── main.js         # App entry point
│   ├── config.js       # Configuration
│   ├── init3DView.js   # 3D setup
│   ├── pipeline/       # Pipeline editor (flowchart, nodes, connections)
│   ├── feeds/          # Camera feed streaming
│   ├── settings/       # Settings UI
│   ├── ui/             # UI utilities
│   └── dropdown/       # Dropdown components
├── html/
│   ├── tabs/           # Tab page templates
│   └── partials/       # Reusable partials
├── css/                # Component styles
└── static/             # Build output (Flask serves from here)
```

## API Integration

Backend base URL hardcoded: `http://localhost:5001/`

Key endpoints:
- `GET /get-settings` - Retrieve app settings
- `POST /save-settings` - Update settings
- `GET /camera/<name>` - MJPEG stream
- `GET /get-available-cameras` - Camera list
- `GET /get-pipeline-objects` - Pipeline configuration
- POST `/restart-backend` - Trigger restart

## Code Style for Frontend

- **ES6 modules**: Standard JavaScript import/export
- **Prettier formatting**: tabWidth: 4 (from package.json)
- **Type hints**: Use JSDoc comments for public functions
- **Descriptive names**: No single-letter variables
- **Architectural comments**: Encouraged for complex blocks and non-obvious design decisions; avoid line-by-line comments

## Common Gotchas

1. **Static file serving**: Changes to `src/webui/static/` won't rebuild—edit source in `src/webui/js/`, `css/`, `html/`
2. **Vite root confusion**: Relative paths in config are relative to `src/webui/`, not project root
3. **Build required for Flask**: Without `npm run build`, Flask serves only API endpoints
4. **Three.js aliases**: Must use alias names (three, OrbitControls, etc.), not full paths
5. **Port conflicts**: Vite dev server (5173) and Flask (5001) must both be available
6. **Partial directories**: Both tabs and partials directories are included—choose based on reusability
