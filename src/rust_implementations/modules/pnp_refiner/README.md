# PnP refiner

`pnp_refiner.PnpRefiner` refines a camera-from-object pose with Levenberg-Marquardt. It stands in for OpenCV's `solvePnPRefineLM` in the PnP camera localization op, which is several times slower on the few dozen points of an AprilTag solve.

```python
from pnp_refiner import PnpRefiner

refiner = PnpRefiner(camera_matrix, distortion)  # 3x3 and 0, 4, 5 or 8 float64 coefficients
ok = refiner.refine(object_points, image_points, rvec, tvec, max_iterations, epsilon)
```

`refine()` updates the float64 `rvec` and `tvec` arrays in place, like OpenCV's in/out arrays. Every attempted step, accepted or rejected, counts toward `max_iterations`. Refinement stops early once a step's norm falls below `epsilon`. It returns `False`, leaving the pose unchanged, when the starting pose puts a point behind the camera. The constructor raises `ValueError` for other distortion models, such as thin-prism or tilted.

`PnpLocalization` falls back to OpenCV's refiner when this module is not built, its constructor rejects the camera, or `refine()` returns `False`.

Rotation steps are multiplicative (`R <- exp(delta) R`), while OpenCV steps the rotation vector additively. Both converge to the same minimum. Poses can differ slightly when the iteration limit stops refinement before convergence.

Build from `src/rust_implementations` with `uv run python build.py pnp_refiner`. Run the Rust unit tests with `cargo test --lib` in this directory.
