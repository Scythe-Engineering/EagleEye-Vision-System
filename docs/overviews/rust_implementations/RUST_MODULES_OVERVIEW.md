# Rust module and build index

Rust Python extensions live under `src/rust_implementations/` and use PyO3 with Maturin.

## Modules

- [`pose_outlier_filter`](../../../src/rust_implementations/modules/pose_outlier_filter/README.md) filters pose outliers using recent accepted poses.
- [`temporal_acceleration`](../../../src/rust_implementations/modules/temporal_acceleration/README.md) computes predicted image regions from pose and AprilTag geometry.

Each module has its own `Cargo.toml`, `Cargo.lock`, Rust source, and README. `module_template/` is the starting layout for another extension.

## Build

Run from the repository root:

```bash
python src/rust_implementations/build.py
python src/rust_implementations/build.py --all
python src/rust_implementations/build.py pose_outlier_filter
python src/rust_implementations/build.py --clean
```

The build script discovers directories under `modules/` that contain `Cargo.toml`. It hashes each manifest and its Rust sources, builds changed modules with `python -m maturin develop`, tests the Python import, and records hashes in `.build_cache.json`.
