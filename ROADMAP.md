# EagleEye improvement roadmap

Remaining gaps identified in the 2026-08-22 competitive audit against Limelight 4 and
PhotonVision, minus the two already done (PolyForm NC relicense, CI-built Pi image).
Ordered by adoption impact.

## Code

### 1. Merge and extend the robot-side library
`feature/robot-localization-library` (currently local-only — push it) has the copy-in
`EagleEyeCamera.java` with the timestamp-joined pose+meta contract.
**Recommendation:** merge it, add a documented example wiring it into WPILib's
`SwerveDrivePoseEstimator`, then add a minimal sim mode (publish poses from a supplied
trajectory over NT) so robot code can be developed without hardware. PhotonLib's sim
support is the gap that costs the most with strong teams.

### 2. Gyro-constrained localization (MegaTag2 equivalent)
No fused localization mode; single-tag and long-range poses are noisier than both
competitors.
**Recommendation:** let PnP accept robot yaw as an optional input (a
`get_networktables_value` node already exists to fetch it) and run a yaw-constrained
solve. Ship it in the default preset once stable.

### 3. Complete default pipelines
`2026_apriltag_starter` is intentionally incomplete (blank fields, no NT publish) and no
object-detection preset exists.
**Recommendation:** ship two complete presets on every install — AprilTag localization and
object detection — pre-wired through Publish To NetworkTables with the bundled season map
defaulted, so the only required act is picking a camera from the dropdown. Target: a pose
on NT without ever opening the graph editor.

### 4. Benchmark results
Every performance claim (temporal acceleration, MX3 throughput) is currently unnumbered.
**Recommendation:** measure on a Pi 5: AprilTag fps with/without temporal acceleration at
a fixed resolution, multi-camera scaling, and object-detection fps on MX3 vs CPU. Publish
in the docs and README; these numbers are the marketing.

### 5. Ship a trained game-piece model
Teams must train their own; Photon ships a 2026 FUEL model, Limelight is drag-and-drop.
**Recommendation:** convert one 2026 model for CPU and MX3, bundle it in the model
library, and document the `eagleeye_training` workflow as a user-guide page.

### 6. Fold NT publishing into pose output (or make the split obvious)
"Robot Pose Output does not publish to NetworkTables" is documented as a thing people get
wrong — a design smell.
**Recommendation:** either add an optional NT key setting to the output node or rename
nodes so the 3D-view/NT split is self-evident.

### 7. First-boot hardening for the image
The image ships user/password `eagleeye`/`eagleeye`.
**Recommendation:** force a password change on first login or document Raspberry Pi
Imager's credential provisioning as the supported path; verify the image boots on real
hardware (Debian 13/trixie base is currently untested — the installer warns).

### 8. Driver-feed streaming
Frames only exist in the EagleEye web UI; both competitors offer a driver camera stream.
**Recommendation:** add an MJPEG endpoint per camera consumable by Shuffleboard/Elastic.

### 9. Camera compatibility guidance
No supported-camera list; no global-shutter guidance.
**Recommendation:** maintain a short tested-cameras table in the docs and recommend
global-shutter sensors for tag detection.

## Docs website (EagleEye-Docs)

### 10. Describe the real configuration UX
`pipeline-setup.md` and `temporal-acceleration.md` show raw `{project_root}/...` paths to
"set", implying manual typing. The UI actually uses camera dropdowns and a Manage/upload
popup for files.
**Recommendation:** rewrite those tables to "choose from the dropdown / click Manage to
upload", with a screenshot of the file manager.

### 11. Update install docs for the flashable image
The guide's install path is still SSH + curl script.
**Recommendation:** make "flash the release image" the primary path (Imager screenshots,
`eagleeye.local:5001`), demote the script install to an advanced/alternative page.

### 12. License page
Docs don't mention licensing.
**Recommendation:** add a short page: free for all teams and noncommercial use forever;
commercial users contact the maintainer. Plain language beats legalese here.

### 13. Robot integration guide
Docs end at "publish to NT"; the guide notes robot-side E2E was never tested.
**Recommendation:** once the robot library merges, add a page with the Java example,
timestamp/latency explanation, and a tested roboRIO round-trip (then remove the
"not tested" caveat).

### 14. Benchmarks page
**Recommendation:** publish the numbers from item 4 with methodology (board, camera,
resolution, commit) so they're reproducible and defensible on Chief Delphi.

### 15. Community entry points
No Discord/Chief Delphi presence, single maintainer.
**Recommendation:** open a Discord, post a Chief Delphi build thread with benchmarks, and
recruit 2–3 pilot teams for the 2027 season. Add a "Get help" page linking these.
