# Rust extensions

Build the PyO3 extension modules with Rust, `maturin`, and the project's Python environment.

```bash
uv run python build.py          # build changed modules
uv run python build.py --all    # rebuild every module
uv run python build.py --list   # list modules
uv run python build.py --clean  # remove build artifacts and the build cache
uv run python build.py module_name
```

`build.py` runs `maturin develop` for each module and verifies that Python can import it. It rebuilds a module when its `Cargo.toml` or Rust source changes. Run `uv sync` first if `maturin` is not installed.

Each module lives in `modules/<name>/` and needs a `Cargo.toml` plus `src/lib.rs`. The crate and Python extension names must match the module directory name.
