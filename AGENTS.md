# Agent notes

## Commands

- Tests: `uv run pytest`. The first run builds the Rust modules in `src/rust_implementations/modules/`. Later runs rebuild only modules whose sources changed.
- Python lint: `uv run ruff check .`
- Rust format: run `cargo fmt` inside each crate under `src/rust_implementations/modules/`.
- CI runs the ruff check, `cargo fmt --check` and pytest.
- Rebuild one Rust module: `uv run python src/rust_implementations/build.py <module>`.
- Run Python through `uv run`. The system `python3` doesn't have the project's packages.

## Benchmarks

`uv run python -m benchmarks.temporal_experiments --config benchmarks/configs/temporal_rust.json --output <new dir>` replays the cached clips through the Rust detector. [benchmarks/README.md](benchmarks/README.md) covers configs and comparing two builds.

## Devices

Pi access and deploy steps are in `CLAUDE.local.md`, which git ignores. If it's missing, ask the user.
