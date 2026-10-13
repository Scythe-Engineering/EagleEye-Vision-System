# WPILib 2026 and 2027 alpha-7 support

Implemented and validated October 13, 2026. EagleEye now supports **WPILib 2026 and 2027.0.0-alpha-7** through version-specific Java SDKs and a Python compatibility boundary. The default coprocessor environment remains 2026; the alpha Python environment is an explicit, platform-gated choice. A 2026 coprocessor can also communicate with an alpha-7 robot over NT4.

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

No physical Systemcore/roboRIO deployment, Driver Station alpha-7/controller integration, camera exposure timestamp verification, radio/network field test, or dashboard GUI test was performed for this migration. No Pi image build/flash/boot or long-duration clock, reconnection, or load test was performed. The eight Python skips cover optional hardware/model, Blender, pipeline configuration, and Linux ABI checks. Passing desktop/loopback checks establish software behavior within that scope, not measured localization accuracy or field certification.

A separate macOS probe exposed a **pre-existing WPILib 2026.2.1 native JNI detection-queue defect**: `readQueueStringArray` uses an `Object[]` constructor signature where `TimestampedStringArray` expects `String[]`, causing a native crash. The passing 2026 pose/meta contract suite does not validate that failing detection-queue path. Alpha-7's real detection-queue regression passes. This migration does not fix the upstream 2026 native defect.

The bundled field assets remain the 2026 game assets. No future 2027 competition layout or game-specific behavior is assumed.

## Primary upstream references

- [Alpha-7 release](https://github.com/wpilibsuite/allwpilib/releases/tag/v2027.0.0-alpha-7) and [inspected commit](https://github.com/wpilibsuite/allwpilib/commit/83df3ee3ce1e892f76970ec21c8efdc00a104d4a).
- [Native JNI timestamp contract](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/generated/main/java/org/wpilib/networktables/NetworkTablesJNI.java), [wire encoder](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/native/cpp/net/WireEncoder.cpp), and [wire decoder](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/native/cpp/net/WireDecoder.cpp).
- [DriverStationErrors](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/wpilibj/src/main/java/org/wpilib/driverstation/DriverStationErrors.java) and [publisher/subscriber options](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/ntcore/src/main/java/org/wpilib/networktables/PubSubOption.java).
- [Geometry](https://github.com/wpilibsuite/allwpilib/tree/v2027.0.0-alpha-7/wpimath/src/main/java/org/wpilib/math/geometry), [field API](https://github.com/wpilibsuite/allwpilib/tree/v2027.0.0-alpha-7/fields/src/main/java/org/wpilib/fields), and [telemetry migration](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/telemetry/doc/telemetry.md#migration-from-wpilib-2026).
- [Alpha build requirements](https://github.com/wpilibsuite/allwpilib/blob/v2027.0.0-alpha-7/build.gradle) and [tagged Java project template](https://github.com/wpilibsuite/vscode-wpilib/tree/v2027.0.0-alpha-7/vscode-wpilib/resources/gradle).
- [pyntcore alpha-7 artifacts](https://pypi.org/project/pyntcore/2027.0.0a7/) and [robotpy-wpimath alpha-7 artifacts](https://pypi.org/project/robotpy-wpimath/2027.0.0a7/).
