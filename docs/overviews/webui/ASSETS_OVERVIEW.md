# WebUI assets

Assets live in `src/webui/assets/` and are served below `/assets/<path>`.

- `robots/` contains GLB robot models. A model may have a sibling `.metadata.json` file for scale data. The asset manager also exposes robot upload, deletion, scale, listing, and `/get-robot-file/<filename>` routes.
- `fields/<year>/field_files/` contains field GLB files. `game_pieces/` contains optional game-piece GLB files. The field asset API manages listings, uploads, deletion, and scale.
- `apriltags/` contains WebP images named `tag36_11_00000.webp` through `tag36_11_00040.webp`. The older `/src/webui/assets/apriltags/<filename>` route also serves this directory.
- `delete.svg` and `settings.svg` are controls used by the pipeline and settings pages.
- `background.webp`, `favicon.ico`, and `no_image.png` supply the page background, icon, and missing-camera image.

The checked-in field directories currently cover 2025 GLB assets and a 2026 field GLB plus AprilTag map. Keep season files under their year rather than adding year-specific paths to JavaScript. Filenames may contain spaces and punctuation; pass them as URL path values rather than rewriting them.

Draco decoder files are served from `/draco/<path>`. The server prepares generated Draco assets at startup, so generated output does not belong in this guide or the source asset list.
