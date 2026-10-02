# Native level-robot 2D PnP

Import `PnpLocalization2D` from `pnp_localization_2d` after building the extension.

```python
solver = PnpLocalization2D(
    camera_matrix,           # 9 finite row-major floats; positive fx/fy
    distortion_coefficients, # 0/4/5/8/12/14 finite floats, OpenCV ordering
    apriltag_ids,            # unique signed 64-bit IDs
    apriltag_corners,        # 12 finite row-major floats per ID (4x3)
)
result = solver.solve(
    capture_us=capture_us,
    gyro_samples=gyro_samples,
    detections=detections,
    mounting_transform=mounting_transform, # row-major robot-from-camera 4x4, or None
    refinement_iterations=10,
    gyro_max_gap_us=100_000,
    gyro_nearest_us=20_000,
)
```

Geometry/calibration are immutable: construct a new solver to change them.
Mounting is supplied on every solve. Detections are a list/tuple of objects with
`tag_id`/`corners` attributes or dictionaries with those keys; corners are 4x2.
Gyro samples are a nonempty list of exact two-key `timestamp_us`/`value`
dictionaries. Timestamps are exact Python ints in `[2, 2**63-1]`; only selected
values must be finite numeric yaw radians. Last duplicate timestamp wins.

Outputs are exactly `camera_pose` (row-major 16 floats), `pose_meta`
(`[tag_count, mean_tag_distance_m, mean_corner_pixel_error]`) and `diagnostics`.
Rejections return `None` for pose/meta and a diagnostic reason. Optional
`align_heading(samples, capture_us, max_gap_us, nearest_us)` returns yaw and
alignment diagnostics, raising `ValueError` on rejected alignment.

Build/check this crate only, keeping generated artifacts outside the repository:

```sh
export CARGO_TARGET_DIR=/tmp/eagleeye-rust-2d-pnp-work/cargo-target
manifest=src/rust_implementations/modules/pnp_localization_2d/Cargo.toml
cargo fmt --manifest-path "$manifest" --check
cargo clippy --manifest-path "$manifest" --all-targets -- -D warnings
cargo test --manifest-path "$manifest"
maturin build --manifest-path "$manifest" --release --out /tmp/eagleeye-rust-2d-pnp-work/wheels
```

Default `extension-module` matches the repository's native modules. The pure
Rust unit tests run with defaults (no embedded interpreter required); use
`--no-default-features` only when embedding/linking against available libpython.
All solve math is native binary64, using nalgebra SVD rather than normal
equations and libm IEEE remainder for yaw. OpenCV-compatible calibration supports
rational distortion, thin prism and sensor tilt, with five inverse-distortion
iterations. Parsing never calls Python cv2/numpy for numerical calculations;
the numerical solve releases Python's thread-state after parsing.
