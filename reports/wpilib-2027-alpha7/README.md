# WPILib 2026 and 2027 alpha-7 support

Implemented and validated October 13, 2026; fresh-install, simulator, and live-runtime checks added October 14, 2026 (see [that section](#fresh-install-and-live-runtime-validation-october-14-2026)). EagleEye now supports **WPILib 2026 and 2027.0.0-alpha-7** through version-specific Java SDKs and a Python compatibility boundary. The default coprocessor environment remains 2026; the alpha Python environment is an explicit, platform-gated choice. A 2026 coprocessor can also communicate with an alpha-7 robot over NT4.

This report records the original audit findings, the independent Opus 5.5 review corrections, the adopted implementation, and final software validation. It supersedes the initial audit's assessment of the unchanged source. [The historical review and adopted-decisions appendix](opus-review-plan.md) retain the review context.

## Release and source scope

The target is the published [WPILib v2027.0.0-alpha-7 release](https://github.com/wpilibsuite/allwpilib/releases/tag/v2027.0.0-alpha-7), published September 1, 2026. The inspected allwpilib tag resolves to **`83df3ee3ce1e892f76970ec21c8efdc00a104d4a`**. Findings were checked against that tagged source and actual released Maven/native and RobotPy artifacts, rather than inferred from upstream `main`.

The original EagleEye audit began at `2fb666c424e205c309062416f05d8896bb39e31e`; its corresponding EagleEye-Docs source was `177b0367d189fb4f3b6245697de2e3d8f5d1e0a6`. The implementation retains Java 2026.2.1 and Python 2026.2.2 support and adds the exact alpha-7 target, not an open-ended promise for later 2027 releases.

## Original failures and review corrections

| Area | Failure in the original source | Implemented resolution |
|---|---|---|
| Java imports and options | `edu.wpi.first.*` types and boolean option factories did not compile against alpha-7. A prefix-only prototype still had eight compiler errors. | Separate `library/java-2027/` sources use the actual `org.wpilib.*` locations and option constants. Existing `library/java/` remains the 2026 entrypoint. |
| Java timestamps | Interpreting native nanoseconds as microseconds made the 0.5-second freshness gate effectively 0.5 ms and observation seconds 1,000 times too large. Retention and warning throttling had the same defect. | Every native alpha timestamp conversion uses `1e9`; unit-bearing identifiers and tests use nanoseconds. The 2026 SDK keeps microsecond conversions. |
| Driver Station warnings | `DriverStation.reportWarning` moved. The original audit incorrectly concluded that an Alert was mandatory; its diagnostic prototype used stderr. | The review verified `DriverStationErrors.reportWarning`, which reaches the Driver Station/HAL reporting path. Production alpha code uses it and retains five-second throttling. |
| Java simulator and example | Field/tag APIs, optional planar bearings, telemetry, estimator constructors, and Java/build tooling changed. | Alpha simulator handles an absent bearing, uses `Field`/`Fields`, and the example uses Telemetry, GradleRIO alpha-7, Gradle 9.4.1, and JDK 25. The 2026 example remains available. |
| Python imports and calls | Geometry moved out of `wpimath.geometry`; NT methods/options/accessors changed to snake_case. Startup failed and some monitoring/setup paths silently degraded. | `src/utils/wpilib_compat.py` selects binding shape once and owns native calls, geometry exports, options, accessors, and clock conversion. |
| Pipeline commands and reads | The pipeline active command silently defaulted to enabled after a missing `getBoolean` method. Numeric reads, flush, and benchmark replay also used removed methods. | All affected calls use the adapter. Real-binding tests verify disable/enable, missing values, numeric reads, flush behavior, setup topics, and status. |
| Python clock and publication | `_now()` still existed but returned nanoseconds. Offset arithmetic mixed rates; alpha publisher timestamps also require nanoseconds. | `TimingMetadata.capture_nt_us` stays in microseconds. Alpha `_now()` is divided by 1,000 and the publish boundary multiplies capture microseconds by 1,000. Startup verifies clock rate before using the cached offset. |
| Packaging/platform | Unbounded package requirements could select a later incompatible stable release. Linux alpha wheels require glibc 2.39, beyond Debian 12/Bookworm. | Mutually exclusive dependency groups, bounded 2026 requirements, exact alpha pins, installer glibc gating, persisted selection, and versioned CI checks. |

The review also corrected the clock description: alpha NT time uses a monotonic clock, rather than the desktop 2026 Unix-anchored epoch. HAL simulation can provide simulated time. Measuring the NT-to-monotonic offset remains appropriate; its magnitude is not a correctness criterion.

For robot consumers, dividing alpha received timestamps by `1e9` matches the **default** `Timer.getTimestamp()` time base. A custom `RobotController.setTimeSource` must preserve that base or translate observations before fusion. This assumption is documented in the SDK and integration guidance.

## Adopted implementation and operating choices

- **Java:** `library/java/` and `library/examples/localization-sim/` remain 2026. `library/java-2027/` and `library/examples/localization-sim-2027/` provide alpha-7. `library/tools/check_java_sdk_parity.py` permits only explicit API/time-unit substitutions between SDK trees, and CI runs both projects with their matching JDKs.
- **Python:** `wpilib2026` is a default dependency group alongside development dependencies. `wpilib2027-alpha7` is mutually exclusive and pins both native packages to `2027.0.0a7`. Adapter import rejects unsupported versions and mixed binding/version families before runtime initialization.
- **Installation:** `bash install.sh --wpilib 2026|2027-alpha7` chooses the environment. `.wpilib-variant` preserves the choice for repair/upgrade. The Linux alpha installer requires glibc ≥ 2.39 and gives Debian 13 / Raspberry Pi OS Trixie or Ubuntu 24.04 guidance when the gate fails; it does not attempt a source build.
- **Timing and transport:** Internal capture timing, NT4 wire microseconds, table/topic names, payload layouts, NWU coordinates, and pose/meta joins remain unchanged. Publishing scales only at the alpha native boundary. Both values retain the same capture timestamp, and completed branches flush together.
- **Controller setup:** The UI and docs refer to the robot controller (roboRIO/Systemcore) or simulation. EagleEye adds no Systemcore-specific hostname, IP default, or team-number mode. The [Systemcore user guide](#systemcore-documentation) reports the addresses that WPILib publishes and tells users to choose one the coprocessor can reach.
- **Pi image:** The image remains a 2026 coprocessor build. Its build metadata records the actual base OS, glibc, and WPILib variant. The existing configurable/latest base-image URL remains; this change does not claim a pinned Bookworm image.
- **Documentation:** The root README, SDK/example guidance, validation notes, and EagleEye-Docs now distinguish both versions, installation choices, API spellings, native timestamp units, default robot time source, and simulation telemetry paths. EagleEye-Docs also has a dedicated Systemcore guide.

Python environment commands:

```sh
# Default 2026 environment
uv sync --locked
uv run --no-sync pytest

# Explicit alpha-7 environment
uv sync --locked --no-group wpilib2026 --group wpilib2027-alpha7 \
  --no-build-package pyntcore --no-build-package robotpy-wpimath
uv run --no-sync pytest
```

These commands allow unrelated project dependencies to build when needed, while requiring native WPILib wheels for alpha. The Ubuntu 24.04 x86 CI matrix can use blanket `--no-build`, where the locked dependencies have wheels.

## Final software validation

These results apply to the completed migration and use real native bindings/artifacts. They are distinct from the smaller exploratory audit checks below.

| Check | Final result | Scope |
|---|---|---|
| Python 2026 full suite | **411 passed, 8 skipped** | Coherent locked 2026 environment, normal Rust extension build/import checks. |
| Python alpha-7 full suite | **411 passed, 8 skipped** | Coherent project environment with real `pyntcore` and `robotpy-wpimath` `2027.0.0a7`, normal Rust checks. |
| Java 2026 | **13 contract tests passed; Gradle test/build passed** | Real 2026 GradleRIO artifacts, JDK 17, desktop HAL. |
| Java alpha-7 | **15 contract tests passed; Gradle test/build passed** | Real alpha-7 artifacts, Gradle 9.4.1, JDK 25, desktop HAL. |
| Java SDK parity | **Passed** | Both SDK source trees match after the explicit migration substitutions. |
| Frontend | **40 tests passed; production build passed** | WebUI regression suite and asset build. |
| EagleEye-Docs | **Production build passed** | Versioned integration and setup documentation. |

New Python regressions independently measure clock scale and conversion accuracy, reject a wrong unit scale, reject unsupported/mixed installations, exercise actual NT pipeline disable and double reads, and check real monitoring/setup topic discovery. A real client/server loopback test publishes repeated identical pose/meta payloads with distinct capture times, verifies all frames arrive, verifies exact pose/meta timestamp equality, and compares received times with captures after unit normalization.

Java regressions check a 20 ms-old observation is accepted, ±600 ms data is rejected, seconds agree with the default robot clock within 1 ms, and Driver Station warning throttling retains the five-second interval. Alpha-specific cases cover detection freshness/retention and the simulator's absent planar bearing.

## Final independent review

The final Opus review found no blocking migration gaps. Its findings and the fixes:

- **Clock-rate check.** A single 5 ms check was sensitive to the scheduler: a pause between clock reads could reject a valid installation. The check now uses 50 ms windows, accepts any valid attempt across up to three attempts, and fails only when all attempts report an invalid rate. A deterministic regression injects a skewed first attempt followed by a valid retry. The existing wrong-unit regression still rejects an actual rate mismatch.
- **Typing and diagnostics.** Version-selected option keyword dictionaries resolve native-stub typing differences. Startup diagnostics distinguish unmanaged developer environments from persisted installer selections and still enforce managed selections. A real-binding subprocess test verifies version reporting, and its code string resolves the flagged lint issue.
- **CI.** Java CI path filters were removed so root-level changes that affect parity run the contract checks. The parity script's executable bit was corrected.

Both full Python suites were rerun after these fixes (**411 passed, 8 skipped** each). Scoped mypy checks passed for the compatibility and timing modules in both environments.

## Fresh-install and live-runtime validation (October 14, 2026)

The sections above record the October 13 audit and implementation. This section records a second pass on October 14 on macOS 26.6.2 (arm64), run against commit `b652eee` plus uncommitted changes to the alpha-7 example (`settings.gradle`, `.wpilib/wpilib_preferences.json`, `.gitignore`). The raw logs and probes were kept in a temporary directory outside the repository and are not committed. A compressed copy (`wpilib-alpha7-validation-20261014.tar.gz`, SHA-256 `642255a4960dc1c551efc19c2e764f496f3dd11630ad898f490bf7cfd8182a4c`) was saved in this machine's user cache folder. It leaves out the installer DMG and the Gradle homes, and a local cache is not a durable record, so treat these results as unreproducible from the repository alone.

**What changed since October 13.** The October 13 work was software validation: unit and contract tests and probes against the alpha-7 jars, with no GUI, reconnection, or load testing. October 14 added a fresh official install and build, the editor and simulator, Glass, the web interface (including a disconnect and reconnect), and the loopback runtime probes below. The uncommitted changes from this pass touch only the alpha-7 example project and the documentation, not production code.

### Fresh install and build

- **Install.** The official macOS arm64 WPILib alpha-7 installer (DMG SHA-256 `2bcdd506537dc4efea058285f2065c30d413ba2ae56fc8d5b4cbbe9286f11a2e`) was run through its own window, choosing **Everything**, for the current user, with VS Code downloaded. It created `~/.wpilib/2027_alpha7` with JDK 25.0.4.1, VS Code 1.134.0, and the WPILib extension 2027.0.0-alpha-7.
- **Defect found.** The example's `settings.gradle` looked for the GradleRIO plugin under `~/wpilib/2027/maven`, which this installer does not create. With a new Gradle home and `--offline`, plugin lookup failed for `org.wpilib.GradleRIO:2027.0.0-alpha-7`. The same original project built when online, because Gradle fetched the plugin from the plugin portal.
- **Fix.** `settings.gradle` is now byte-identical to the vscode-wpilib `v2027.0.0-alpha-7` template. It searches the installer's `2027_alpha7` folder first, then local Maven, then the plugin portal. The macOS folder is `~/.wpilib/2027_alpha7` (`~/wpilib/2027_alpha7` is a link to it); the Linux and Windows folders are listed in the [example README](../../library/examples/localization-sim-2027/README.md). Only the macOS install was run. The other two folders come from the same template.
- **Result.** With the fixed file, a clean offline `./gradlew test build` using a new Gradle home passed all 15 contract tests. The Gradle wrapper had been downloaded online first, and `--offline` does not cover that download, so this is not a build on a blank machine with no network.
- **Preferences.** `projectYear` in `.wpilib/wpilib_preferences.json` changed from `2027` to `2027_alpha7`, and `bin/` is now ignored.
- **Other checks.** The 41 targeted Python tests (compatibility, timing, publishing, startup checks) passed, and the frontend production build passed. The full Python suites were not rerun on October 14.

### Editor and simulator (native windows)

- **Project import.** With `projectYear` set to `2027`, the WPILib extension offered to import the example when it was opened. With `2027_alpha7` and a **Developer: Reload Window**, the prompt no longer appeared and Java support finished loading.
- **Simulation.** The extension's **WPILib: Simulate Robot Code** command ran the `simulateExternalJava` task successfully, and **Sim GUI** was selected. A JDK 25 process running `frc.robot.Main` served NetworkTables on port 5810, and the extension log showed the launch arguments and the Sim GUI extension. The command-line logs show the Sim GUI extension loaded and "Simulator GUI Initialized!"; the Java simulator window itself was not seen (the native window inventory did not list it), so the evidence is that the extension initialized, not what it displayed. The IDE launch, the NetworkTables server, and Glass were each observed directly.
- **Command line.** In alpha-7 the GradleRIO Java simulation task is `simulateExternalJava`; the plugin sources define no task named `simulateJava`, but Gradle accepts `./gradlew simulateJava` as an abbreviation (a `--dry-run` resolved it to `simulateExternalJava`). That task only builds and writes `build/sim/java.json`, the launch description the extension reads. It does not start the robot program. Running `./gradlew run` with the Sim GUI extension in `HALSIM_EXTENSIONS` loaded the extension and listened on 5810; the process was then terminated from outside, which is why Gradle printed exit value 143.
- **Accidental deploy.** While trying to stop the simulator from the editor with a keyboard shortcut, a WPILib deploy for team 0 started. No robot was connected, discovery timed out, and nothing was deployed. The owned simulator process was then stopped directly.
- **Glass.** A fresh Glass window connected to `127.0.0.1:5810`. `Telemetry/EagleEye/Accepted` rose from 3,667 to 4,301 while it was open. The `GroundTruth` and `Robot` topics were present as `Pose2d[]` structs, with a `.type` of `Field2d`. Whether Glass drew the field widget was not checked.

### EagleEye backend and web interface

The default 2026 backend served the web interface with an empty camera list (the first-boot wizard was skipped), live system information, and the pipeline editor. Settings reported **Disconnected** before the simulator started and **Connected** afterward. The 2026 backend was then stopped and the full backend was started from an existing isolated Python environment with the alpha-7 packages (`2027.0.0a7`). After a reload in Safari, Settings reported **Connected**.

On the alpha-7 backend, the full sequence was then observed in Safari (native accessibility readout): Settings **Connected**, then the simulator process was terminated and Settings showed **Disconnected**, then the simulator was restarted with `./gradlew run` and Settings showed **Connected** again. The backend log recorded NT4 connected, disconnected ("remote end closed connection"), and connected again, and the restarted simulator's log shows both `glass@1` and `EagleEye@2` connecting. This is one pass of one sequence on the alpha-7 backend, and the restart was from the command line rather than the editor.

No pipelines were configured and no camera was authorized, so this did not run a camera through a pipeline. The publisher and the pipeline enable/disable command were covered only by the separate runtime probes below. The backend did try to open a camera and logged an OpenCV authorization failure.

### Python-to-Java runtime probes

These probes run Python production code against the checked-in alpha-7 Java SDK on one computer over NetworkTables 4 on a nonstandard port. They do not start the backend, a camera, a GUI, a real Driver Station, or any controller hardware. The 50 ms difference between capture and publication is synthetic, so the reported ages do not measure camera exposure delay.

| Run | Result |
| --- | --- |
| Python 2026.2.2 publisher to alpha-7 Java SDK | 540 of 540 frames accepted |
| Python 2027.0.0a7 publisher to alpha-7 Java SDK | 540 of 540 frames accepted |
| Alpha-7 Java SDK polled every 20 ms, 90 Hz publisher (short run) | 540 of 540 frames accepted; largest raw burst 3 |
| 60 seconds with no catch-up bursts, polled every 20 ms | 4,420 of 4,420 frames accepted; largest raw burst 4 |
| About 5 minutes (311.8 s), 90 Hz target, catch-up after delays, polled every 3 ms | **26,974 of 27,000 accepted. This run failed its exact-count check.** |

- **Short runs.** Each of the first two runs sent three blocks of 180 frames. The accepted counts are cumulative: 180 before the Java side disabled the pipeline, still 180 during the disable, 360 after re-enable, and 540 after a server restart (180 per block, not 360 in one block). Pose X/Y/yaw stayed at 2/3/0.5. Three stale and three future capture times were rejected, and the real Python pipeline command gate read false while disabled and true after re-enable.
- **No-catch-up control.** The publisher set each deadline from its actual send time, so it never sent frames back to back to make up lost time. It achieved 73.67 frames per second, not 90 (minimum gap 11.1 ms, median 13.9 ms, maximum 42.1 ms), so it is not evidence of sustained 90 Hz delivery. Independent raw subscribers also received all 4,420. Its largest raw burst was 4, below the SDK's 20-sample queue. It differs from the five-minute run in duration (60 s against about 312 s), pacing, SDK polling interval (20 ms against 3 ms), and the absence of disable, reconnect, and restart phases, so it does not isolate one cause of the loss in that run.
- **Five-minute run.** The publisher kept its original 90 Hz schedule, so after a delay it sent several frames at once. The first block accepted 8,974 of 9,000. After a re-enable and a deliberate client reconnect, the second block accepted 9,000 of 9,000. After a server restart, the third accepted 9,000 of 9,000. Raw subscribers with 1,024-entry queues received all 27,000 pose and metadata pairs, plus two retained values replayed after the reconnects (which the SDK correctly rejected), and their timestamp sets were identical. Accepted frames were 50 to 288 ms old including the synthetic 50 ms offset, inside the SDK's 500 ms limit.
- **Queue evidence.** The raw subscribers logged two bursts, of 22 and 26 frames, both in the first block, where all 26 missing frames occurred. The SDK keeps 20 samples per topic between polls, so those bursts exceed it. That is consistent with queue overflow, but individual missing frames were not traced through the SDK, so overflow is not proven for each one. The run overlapped the WPILib installation on a heavily loaded computer, so the cause of the pauses is also unknown. A separate check that withheld polling while 90 frames arrived confirmed the SDK keeps exactly 20.
- **Disposition.** The failed run is preserved as a failure; no check was loosened to pass it. The 20-sample limit is unchanged in both the 2026 and alpha-7 SDKs, following the review's recommendation to document it rather than change it. The [library guide](../../library/README.md#poll-every-robot-loop) tells teams to poll every loop.

### Not covered

The VS Code new-project creator was tried, but its window stayed blank under automation and native screenshots were intermittently stale, so project creation is not reported as passing. AdvantageScope and Elastic were not tested. The built-in T3 browser preview timed out twice, so the web interface was exercised through native Safari. Driver Station controls in the Java simulator window, a physical controller, and the Driver Station itself were not tested.

## Historical interoperability evidence

The **original audit**, before the final compatibility adapter and production Java migration, exercised a 2026 Python client against alpha-7 servers:

1. An alpha-7 Python NT4 server received **10 poses, 10 metadata arrays, and 10 detection arrays**, including repeated identical payloads at distinct capture times. Their timestamp queues matched exactly.
2. An alpha-7 Java NT4 server using the diagnostic migrated SDK accepted **all 10 observations**, preserved X/Y/yaw (2/3/0.5), produced seconds timestamps, and joined a detection to the accepted pose.

This is historical mixed-version transport evidence, not a new final-production mixed Java/hardware run. Tagged source retains the Pose3d struct schema and converts native alpha nanoseconds to wire microseconds and back, explaining the observed interoperability.

The original report, review, raw logs, and diagnostic probes were copied and checksum-verified to an ephemeral local archive before removing raw evidence/probes from the publication tree. That archive is machine-local, ephemeral, and not required to run the committed regression suites.

## Systemcore documentation

The EagleEye-Docs page *Use EagleEye with Systemcore* describes an external-coprocessor setup: EagleEye stays on a Raspberry Pi and connects to the robot program on Systemcore over NT4. It was written from published sources read on October 13, 2026, not from hardware testing:

- The [SystemcoreTesting README](https://github.com/wpilibsuite/SystemcoreTesting) supplies the compatibility table (image 14 and later pairs with WPILib alpha-7 and later), the address table, team-number setup, and first-deployment steps.
- The [alpha-7 release notes](https://github.com/wpilibsuite/allwpilib/releases/tag/v2027.0.0-alpha-7) state that Systemcore image 14 and Driver Station alpha-7 are required.
- WPILib's 2027 [imaging](https://docs.wpilib.org/en/2027/docs/zero-to-robot/step-3/imaging-your-systemcore.html) and [SSH](https://docs.wpilib.org/en/2027/docs/software/systemcore-info/systemcore-ssh.html) pages supply the flashing flow and the address and password details. The 2027 web dashboard page is still a placeholder.
- The [Systemcore OS repository](https://github.com/LimelightVision/systemcore-os-public) shows a Linux-based OS with its own package manager.

Running EagleEye on the Systemcore itself is not supported, and the guide lists no onboard installation commands. It does not guess a Systemcore address: the USB and Wi-Fi access-point addresses are labeled as links the Pi may not be on, and the Ethernet address is read from the controller display. For web flashing it recommends image 12 or later (WPILib's imaging guide says 11, the SystemcoreTesting README says 12) and points older images to flash mode.

## Limits and outstanding validation

No physical Systemcore/roboRIO deployment, Driver Station alpha-7/controller integration, camera exposure timestamp verification, radio/network field test, or camera-to-pipeline run was performed for this migration. No Pi image build/flash/boot was performed, and EagleEye was not installed or run on a Systemcore. The October 13 pass had no GUI, reconnection, or load testing. October 14 added a desktop simulator and web interface check (including one simulator stop, a Disconnected reading, and a restart back to Connected), a five-minute loopback soak that failed its exact-count check, and server-restart and reconnect probes. Those do not replace multi-hour, field-network, or hardware testing. The eight Python skips cover optional hardware/model, Blender, pipeline configuration, and Linux ABI checks. Passing desktop/loopback checks establish software behavior within that scope, not measured localization accuracy or field certification.

A separate macOS probe exposed a **pre-existing WPILib 2026.2.1 native JNI detection-queue defect**: `readQueueStringArray` uses an `Object[]` constructor signature where `TimestampedStringArray` expects `String[]`, causing a native crash. The passing 2026 pose/meta contract suite does not validate that failing detection-queue path. Alpha-7's real detection-queue regression passes. This migration does not fix the upstream 2026 native defect.

The bundled field assets remain the 2026 game assets. No future 2027 competition layout or game-specific behavior is assumed.

## Primary upstream references

- [Alpha-7 release](https://github.com/wpilibsuite/allwpilib/releases/tag/v2027.0.0-alpha-7) and [inspected commit](https://github.com/wpilibsuite/allwpilib/commit/83df3ee3ce1e892f76970ec21c8efdc00a104d4a).
- [Native JNI timestamp contract](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/generated/main/java/org/wpilib/networktables/NetworkTablesJNI.java), [wire encoder](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/native/cpp/net/WireEncoder.cpp), and [wire decoder](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/native/cpp/net/WireDecoder.cpp).
- [DriverStationErrors](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/wpilibj/src/main/java/org/wpilib/driverstation/DriverStationErrors.java) and [publisher/subscriber options](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/java/org/wpilib/networktables/PubSubOption.java).
- [Geometry](https://github.com/wpilibsuite/allwpilib/tree/v2027.0.0-alpha-7/wpimath/src/main/java/org/wpilib/math/geometry), [field API](https://github.com/wpilibsuite/allwpilib/tree/v2027.0.0-alpha-7/fields/src/main/java/org/wpilib/fields), and [telemetry migration](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/telemetry/doc/telemetry.md#migration-from-wpilib-2026).
- [Alpha build requirements](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/build.gradle) and [tagged Java project template](https://github.com/wpilibsuite/vscode-wpilib/tree/v2027.0.0-alpha-7/vscode-wpilib/resources/gradle).
- [pyntcore alpha-7 artifacts](https://pypi.org/project/pyntcore/2027.0.0a7/) and [robotpy-wpimath alpha-7 artifacts](https://pypi.org/project/robotpy-wpimath/2027.0.0a7/).
