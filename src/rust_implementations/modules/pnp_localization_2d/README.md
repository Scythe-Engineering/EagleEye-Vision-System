# Native level-robot 2D PnP

## Production pipeline setup

Build the extension through the existing project workflow:

```sh
uv run python src/rust_implementations/build.py pnp_localization_2d
```

Replace the normal `pnp_camera_localization` node with
`pnp_camera_localization_2d`. Keep the existing detector, minimum-tag gate,
`camera_to_robot_pose` conversion, and temporal feedback connections. Connect a
`get_networktables_value` source's `data` output to the new solver's
`gyro_samples` input. Configure that source with `network_table_key="gyro"`,
`timestamped=true`, and `history_size=256`. Normal 3D PnP remains unchanged.

Publish a native double using `publisher.set(yaw_rad, measurement_nt_us)`, with
NWU yaw radians counterclockwise about field +Z, zero along field +X. Heading
must already match the AprilTag map's field axes, not an arbitrary gyro startup
zero or alliance-mirrored heading. The timestamp is the original measurement
instant in the publisher's local NT clock, even when publication is delayed.
ntcore converts it to the receiver clock; do not use JSON timestamps or manual
offset arithmetic. Use `keepDuplicates=true`, `sendAll=true`, a short period
such as 10 ms, and timely flushing. These settings do not guarantee delivery
within the solver's default 20 ms nearest-sample tolerance.

The reader retains bounded native sample history and rejects client input until
NT clock synchronization is available. Detections must retain their image capture
timestamp. The solver circularly interpolates capture-time heading, allowing a
100 ms bracket or a nearest sample within 20 ms by default. Missing, stale, or
invalid data rejects the current pose with diagnostics, without held poses or
an unconstrained fallback.

Use the same `camera_bus_id` for input, solver, and robot conversion, with the
existing `CameraConfigRegistry`. The model assumes a level robot at field Z=0
and solves robot XY with fixed gyro heading. Mounting pitch/yaw/roll are degrees
and offsets are meters; all six mounting values are fetched live each run.
Intrinsics and map geometry are fixed at construction, so recreate the operation
after editing them. The Python action emits a float64 4x4 field-from-camera EDN
matrix, the three-item quality list below, and explicit diagnostics. Real use
requires calibrated mounting and an actual synchronized gyro publisher.

## Native interface

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
Negative gyro limits or refinement iterations above 100 raise `ValueError`
before measurement validation. Refinement iterations must be in `[0, 100]`;
negative counts fail native unsigned-integer conversion. Valid arguments preserve
measurement/geometry rejection precedence.
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
