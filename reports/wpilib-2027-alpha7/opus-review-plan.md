# WPILib 2027 alpha-7 migration: pre-implementation review and fix plan

Reviewer: Opus 5.5 (medium), 2026-10-13. Review only; no production source changed.
Inputs: `reports/wpilib-2027-alpha7/README.md` (audit), its evidence/probes, tagged upstream
`allwpilib@83df3ee3` (v2027.0.0-alpha-7), released alpha jars, the alpha RobotPy venv
(`pyntcore 2027.0.0a7`), the vscode-wpilib `v2027.0.0-alpha-7` Gradle templates, PyPI metadata,
the EagleEye checkout `2fb666c`, and EagleEye-Docs on the same branch.

## 1. Verdict on the audit

The core conclusions hold: the Java SDK does not compile, NT integer timestamps are now nanoseconds,
the `/1e6` conversions must become `/1e9`, RobotPy renamed its APIs to snake_case and flattened
`wpimath.geometry`, `ntcore._now()` changed units, and a 2026 Python coprocessor interoperates
with an alpha-7 robot. The items below are **corrections** and **omissions** the implementers must
account for.

### 1.1 Corrections

| # | Audit claim | Verified reality | Effect on the plan |
|---|---|---|---|
| C1 | "`DriverStation.reportWarning` is removed; missing-topic warnings need the new alert mechanism." | Released `wpilibj.jar` has `org.wpilib.driverstation.DriverStationErrors.reportWarning(String, boolean)`. It goes through `DriverStationBackend` → `DriverStationJNI.sendError`, the same HAL path as 2026. In simulation, HAL prints it to the console exactly as before. | Use `DriverStationErrors.reportWarning`, keeping the 5 s throttle. Do **not** ship the audit prototype's `System.err.println`, which loses DS delivery on a real robot. `org.wpilib.util.Alert` is an optional addition, not a requirement. |
| C2 | Clock described only as a unit change (µs→ns), using "FPGA time" wording. | The **epoch changed too** (upstream PR #9074, "Always use steady_clock"). In 2026 non-roboRIO builds, `wpi::NowDefault()` = `system_clock` µs captured at start + `steady_clock` delta, so it is Unix-epoch anchored. In alpha-7 it is raw `steady_clock` ns, which is `CLOCK_MONOTONIC` on Linux. On the robot, NT time = `wpi::Now()` = HAL monotonic time. In simulation it is the **simulated** clock (`MockHooks` installs `SetNowImpl(GetMonotonicTime)`), which pauses and steps with sim. | (a) The Python offset-measurement design remains valid; on alpha Linux the offset is ≈0. Tests must not assume a particular offset magnitude. (b) Remove the "FPGA time" wording from the 2027 SDK and docs. The 2027 term is "NT/robot monotonic time, the default `Timer.getTimestamp()` base". |
| C3 | Implicitly: `sample.timestamp / 1e9` always equals the estimator's time base. | `Timer.getTimestamp()` = `RobotController.getTime()`, which defaults to `getMonotonicTime()` but **can be overridden** with `RobotController.setTimeSource`. NT timestamps always use `wpi::Now()`. | Use `/1e9` (exact under the default source, matching the existing contract) and document the assumption in the Javadoc and docs. See unresolved choice U4. |
| C4 | "Wire still µs" (correct), but the API detail is incomplete. | `TimestampedObject`/`TimestampedDoubleArray` expose `timestamp` (local time base, ns) and `serverTime` (ns). Python `Publisher.set(value, time)` takes **ns** in alpha (`0` = now). `WireEncoder` divides by 1000 and `WireDecoder` multiplies by 1000; the client applies the server offset in ns. | The Python publish boundary must send `capture_nt_us * 1000` on alpha. Exact pose/meta timestamp equality survives because both go through the same truncation. |
| C5 | "Monitoring/first-boot use removed methods" (listed as breaks). | These paths are wrapped in `try/except` and will **degrade silently**, not crash. `system_monitor_mixin.py:1031-1046` reports "unavailable". `first_boot_mixin.py:395-400` catches `AttributeError` and returns an empty set, so setup verification says topics are missing. | Tests must exercise these paths against the **real** binding of each version, not only fakes. |
| C6 | Alpha Python can be installed by "regenerating the lock". | Alpha RobotPy wheels exist **only** as `manylinux_2_39` (glibc ≥ 2.39), with no sdist. The 2026 wheels are `manylinux_2_35`. The docs (`prepare-pi.md:8,40`) and `install.sh:4,20,65` target **Debian 12 / Pi OS Bookworm, glibc 2.36**, where alpha wheels **cannot install**. | Alpha Python must be an opt-in, platform-gated variant. The default coprocessor stack stays on 2026. |

### 1.2 Omissions (missing functionality / call sites)

1. **Pipeline enable/disable command is silently ignored on alpha.** `src/config/utils/pipeline.py:982`
   calls `self.network_table.getEntry(cmd).getBoolean(True)` inside `except Exception: return True`.
   With alpha bindings, every pipeline therefore stays permanently active and the robot's
   `.../active` command does nothing. The audit did not mention this.
2. `pipeline.py:712` `getInstance().flush()`: the audit lists this correctly. The alpha equivalent is `get_instance().flush()`.
3. `src/secondary_operations/get_networktables_value.py:33` raises in `run()` on alpha
   (`getEntry`/`getDouble` → `get_entry`/`get_double`).
4. `benchmarks/replay.py:95,221-250` uses `create/startLocal/getTable/stopLocal/destroy`. Benchmark
   replay breaks on alpha.
5. Tests and fakes encode camelCase APIs: `tests/utils/dummy_dependencies.py:118` (`getEntry`),
   `tests/test_web_server_runtime.py:315-323` (`isConnected/getConnections`),
   `tests/test_pipeline_network_flush.py:38-39` (`getInstance`), `tests/test_publish_to_networktables.py:251-260`
   (real ntcore, camelCase options), and `tests/test_position_contract.py:66-70` /
   `test_publish_to_networktables.py:34-69` (`X()`, `rotation()`, `getQuaternion()`). On alpha (verified) these become
   `.x/.y/.z` (float properties), `.rotation()` (still a method), `rotation.x/.y/.z`, and
   `rotation.get_quaternion()` with `.w/.x/.y/.z` properties.
6. Typing-only imports (`web_server.py:30`, `generate_all_pipelines.py:117`, `get_networktables_value.py:12`)
   are fine on both versions, since `ntcore.NetworkTable` and `NetworkTableInstance` still exist.
7. **Version-mixing guard.** Nothing prevents `pyntcore` 2027 with `robotpy-wpimath` 2026, or the reverse,
   across a hand-edited environment. The adapter must detect both and fail loudly on a mismatch.
8. **Unbounded requirements** (`pyproject.toml:10-11`). Pre-releases are not auto-selected, but the first
   **stable** 2027.0.0 will be selected by `uv lock --upgrade` and Dependabot (`.github/dependabot.yml`).
   The 2026 variant needs `<2027` bounds.
9. **Java tooling specifics.** The GradleRIO plugin id is now `org.wpilib.GradleRIO` (`2027.0.0-alpha-7`, not
   `edu.wpi.first.GradleRIO`). The template uses Gradle **9.4.1**, `JavaVersion.VERSION_25`, the
   `application` plugin, and a `systemcore(getTargetTypeClass('SystemCore'))` deploy target. It also needs
   `frcYear = '2027'` in `settings.gradle` and `projectYear: "2027"` in `.wpilib/wpilib_preferences.json`.
   The canonical source is `vscode-wpilib@v2027.0.0-alpha-7/vscode-wpilib/resources/gradle/{java,shared}`.
10. Java contract test (`EagleEyeContractTest.java`): uses `edu.wpi.first.hal.HAL` → `org.wpilib.hardware.hal.HAL`;
    `now - 1000`, `±1_000_000`, `now/1e6` are µs literals and must become ns (`now - 1_000_000`, `±1_000_000_000`, `/1e9`).
11. **CI.** `JavaContract.yaml` hard-codes JDK 17 and one project. `Pytests.yaml` tests only the locked 2026 environment.
    Neither would catch an alpha regression.
12. **Robot-address UX.** The WebUI copy (`settings_tab_content.html:109` "Roborio (or sim) IP Address",
    `wizardState.js:169`) and `library/README.md` say roboRIO. They should say "robot controller
    (roboRIO/Systemcore) or simulation". Optionally, alpha `set_server_team` could support team-number addressing (see U5).
13. Pi image (`ci/build-pi-image.sh:23`) downloads `raspios_lite_arm64_latest`, which may already be Debian 13,
    while the docs and installer claim Debian 12. This inconsistency predates the migration but now matters
    for the glibc gate. The image should pin a base or record its glibc.
14. Pre-existing low-risk item, not a regression: the pose and meta joins rely on both values being encoded
    with the same client→server offset. If a time-sync update lands between the two `set()` calls, the pair
    can diverge by the offset delta. Do not "fix" this in this migration; keep the flush-after-branches design.

## 2. Scope decision

**Support both 2026 and 2027-alpha-7 for the robot SDK and the Python coprocessor. The default stays 2026.**

- **Robot Java:** Java cannot conditionally compile across `edu.wpi.first.*` and `org.wpilib.*`, so ship
  **version-specific source trees** with no shared base. This avoids breaking 2026 teams mid-season and is
  the only way to give alpha testers a working SDK.
- **Python coprocessor:** a **narrow compatibility adapter** that isolates every ntcore and wpimath call. This is cheap
  (~15 call sites), fixes all alpha breaks, and adds no behavior change for 2026 users. The 2026 coprocessor is
  wire-compatible with alpha-7 robots, so the **default stays 2026** for robustness and platform reach (C6).
  Alpha is an explicit opt-in.
- **Do not** convert the backend to an NT server, change topic names or payload layouts, or change NT4 wire units.

## 3. Architecture

### 3.1 Python: `src/utils/wpilib_compat.py` (single owner of binding differences)

```text
API: Literal["2026", "2027"]        # detected once at import
NT_TIME_UNITS_PER_US: int           # 1 (2026) or 1000 (2027)
Pose2d, Pose3d, Quaternion, Rotation2d, Rotation3d, Transform2d, Transform3d,
Translation2d, Translation3d        # re-exported from wpimath.geometry (2026) or wpimath (2027)

default_instance() / create_instance() / destroy_instance(inst)
start_client(inst, identity, server)       # startClient4+setServer | start_client+set_server
start_local(inst) / stop_local(inst)
get_table(inst, name)
table_instance(table)                      # getInstance | get_instance
flush(inst)
is_connected(inst) -> bool; connection_count(inst) -> int; topic_names(inst) -> set[str]
read_bool(table, key, default) / read_double(table, key, default)
pubsub_options(*, keep_duplicates, send_all, periodic, poll_storage=0)
topic(table, kind, key, struct_type=None)  # kind in {"boolean","double","string","boolean_array",
                                           #  "double_array","string_array","struct","struct_array"}
raw_nt_now() -> int                        # ntcore._now() in native units
pose_xyz(pose) / quaternion_wxyz(rot)      # accessor helpers used by tests/contract code only
```

Rules:

- **Detection:** `API = "2027" if hasattr(ntcore.NetworkTableInstance, "get_default") else "2026"`.
  Separately, detect wpimath with `importlib.util.find_spec("wpimath.geometry")`. Raise `ImportError` with
  a clear message if the two disagree, or if neither shape is present. Log `API` and both versions once at
  backend start (`main_backend.py`).
- **Exceptions:** No `except AttributeError` fallbacks inside call paths. Dispatch is decided once, so a
  wrong API raises immediately instead of being swallowed by the existing broad `except` blocks (C5,
  omission 1).
- **Internal timing contract stays microseconds.** `TimingMetadata.capture_nt_us` is unchanged.
  - `timing.now_nt_us()` returns `raw_nt_now() // NT_TIME_UNITS_PER_US`.
  - The publish boundary in `PublishToNetworktables` calls `publisher.set(v, capture_nt_us * NT_TIME_UNITS_PER_US)`,
    exposed as a compat helper `to_nt_publish_time(us)`.
  - Justification: the wire is µs, so no information is lost. This touches 2 functions instead of every
    field, keeps 2026 byte-identical, and keeps benchmarks comparable.
  - Rename nothing in `TimingMetadata`. Update the `now_nt_us` docstring to state the unit normalization and
    the per-version epoch (C2).
- **Startup clock sanity check** in `timing._measure_nt_monotonic_offset_us`: in addition to the median
  offset, take two paired samples ≥5 ms apart and require `Δnt_us / Δmono_us ∈ [0.5, 2]`. Otherwise raise
  `RuntimeError("ntcore clock unit mismatch")`. This turns any future unit change into a loud failure.

Call sites to route through the compat module: `main_backend.py:78-81`,
`publish_to_networktables.py:9-20,249,257-279`, `get_networktables_value.py:33`,
`pipeline.py:712,982`, `system_monitor_mixin.py:1032-1033`, `first_boot_mixin.py:396`, `timing.py:21`,
`benchmarks/replay.py`, tests, and fakes. Fakes should implement the **compat-facing** surface, or the
adapter should dispatch on the fake's shape. Preferred: tests that need real behavior use real ntcore,
and the remaining fakes expose the snake_case 2027 names plus a fake `API` override via monkeypatch.

### 3.2 Python packaging (pyproject / uv.lock / installer)

- Move `pyntcore` and `robotpy-wpimath` out of base `dependencies` into mutually exclusive extras:

  ```toml
  [project.optional-dependencies]
  wpilib2026 = ["pyntcore>=2026.2.2,<2027", "robotpy-wpimath>=2026.2.2,<2027"]
  wpilib2027-alpha7 = ["pyntcore==2027.0.0a7", "robotpy-wpimath==2027.0.0a7"]
  [tool.uv]
  conflicts = [[{ extra = "wpilib2026" }, { extra = "wpilib2027-alpha7" }]]
  ```

  One `uv.lock` resolves both forks. Verify that `uv lock` accepts the explicit pre-release pin without a
  global `prerelease = "allow"`. If not, scope it with `[tool.uv] prerelease = "if-necessary-or-explicit"`.
  Do not use global allow.
- `install.sh`: add `--wpilib 2026|2027-alpha7` (default `2026`), a persisted choice (env or file under
  install dir) so upgrades keep it, and `uv sync --locked --extra wpilib<choice>`.
  - For `2027-alpha7`, check glibc via `ldd --version` or `getconf GNU_LIBC_VERSION` and require ≥ 2.39.
    Otherwise exit with a message naming Debian 13 / Pi OS Trixie as required. Do not attempt a source build.
  - Keep the default path identical in behavior for 2026 users.
- `src/startup/install_check.py`: also report the detected compat `API`.
- CI `Pytests.yaml`: matrix `wpilib: [2026, 2027-alpha7]` running `uv sync --locked --no-build --extra wpilib${{matrix.wpilib}}`
  on ubuntu-24.04 (glibc 2.39, so alpha wheels install). The cache key must include the matrix value.
- `pi-image.yml` / `build-pi-image.sh`: build the default (2026) variant only. Record base OS and glibc in the image manifest.
- `.github/dependabot.yml`: ignore `pyntcore`/`robotpy-*` major bumps, or rely on the `<2027` bound.

### 3.3 Java: version-specific SDK trees

```text
library/java/frc/robot/vision/{EagleEyeCamera,EagleEyeCameraSim}.java        # 2026, path unchanged
library/java-2027/frc/robot/vision/{EagleEyeCamera,EagleEyeCameraSim}.java   # alpha-7
library/examples/localization-sim/        # 2026 (unchanged except docs cross-links)
library/examples/localization-sim-2027/   # new: generated from vscode-wpilib v2027.0.0-alpha-7 java template
library/tools/check_java_sdk_parity.py    # normalizes known 2026<->2027 substitutions and diffs the trees
```

2027 SDK changes. Start from `probes/migrated-*.java`, then:

- Imports: `org.wpilib.math.linalg.{Matrix,VecBuilder}`, `org.wpilib.math.{geometry,numbers}`,
  `org.wpilib.networktables.*`, `org.wpilib.driverstation.DriverStationErrors`.
- `PubSubOption.SEND_ALL`, `KEEP_DUPLICATES`, `pollStorage(n)`.
- Introduce `private static final double NT_TIME_UNITS_PER_SECOND = 1e9;` and use it at every former `/1e6`
  site: 211, 227, 240, 243, 267, 349, 419. Rename `timestampMicros` to `timestampNanos` and
  `nowMicros`/`lastWarningMicros` to `*Nanos`.
- Warnings: `DriverStationErrors.reportWarning(msg, false)` with the existing throttle (C1). Fix the message at
  `:365` to say "robot controller".
- Javadoc (`:30-40`): replace FPGA wording with the NT monotonic/simulated-time semantics and the
  `setTimeSource` caveat (C2, C3).
- Sim: `org.wpilib.fields.Field.loadField(Fields.DEFAULT_FIELD)`, `field.getTags()` with `tag.getPose()`.
  Handle `Translation2d.getAngle()` returning `Optional<Rotation2d>`: an empty value means the camera is
  on the tag, so skip that tag. Constructor Javadoc example updated.
- Example: `org.wpilib.framework.{TimedRobot,RobotBase}`, `org.wpilib.system.Timer.getTimestamp()`,
  `org.wpilib.smartdashboard.Field2d` + `org.wpilib.telemetry.Telemetry.log`.
  `DifferentialDrivePoseEstimator(Rotation2d, double, double, Pose2d[, stdDevs, stdDevs])` (kinematics
  argument removed). `Main` uses `RobotBase.startRobot(Robot::new)`.
  Build: Gradle 9.4.1 wrapper, `org.wpilib.GradleRIO 2027.0.0-alpha-7`, Java 25, SystemCore deploy target,
  `sourceSets.main.java.srcDir('../../java-2027')`, JUnit as in the 2026 example.
- Contract test copy for 2027: `org.wpilib.hardware.hal.HAL`, ns literals (omission 10), plus one new test
  per §4 J2-J5.
- Parity script: it applies an explicit substitution table (imports, option constants, `1e6`→`1e9`,
  `Micros`→`Nanos`, warning call, Field API, Optional bearing) and fails if any **other** line differs.
  Run it in CI so future fixes land in both trees.

CI `JavaContract.yaml`: two jobs.

- `wpilib-2026`: Temurin 17, `library/examples/localization-sim`, existing command.
- `wpilib-2027-alpha7`: Temurin 25, `library/examples/localization-sim-2027`, `./gradlew test build`.
- Both run the parity script, using Python only, no deps.
- Trigger on `library/**` paths as today.

## 4. Tests to add (must fail on the buggy code, not just pass on fixed code)

Python (run in both matrix legs):

- P1 Clock scale: `Δ now_nt_us()` over `time.sleep(0.05)` is within ±20 ms of `Δ time.monotonic_ns()//1000`.
  This fails on an unfixed alpha (×1000).
- P2 `monotonic_ns_to_nt_us(time.monotonic_ns())` is within 2 ms of `now_nt_us()`.
- P3 Real-ntcore loopback: create a server instance on an ephemeral port and a client instance, then publish
  through `PublishToNetworktables` with known `capture_nt_us`. On the server, read the subscriber
  `read_queue()`/`readQueue()` through compat. Assert that pose and meta timestamps are **equal** to each other,
  that `server_ts / NT_TIME_UNITS_PER_US` is within the time-sync tolerance of `capture_nt_us`, and that
  distinct frames with identical payloads arrive separately.
- P4 Pipeline active command: with a real local instance, publishing `False` to the command topic makes
  `_is_enabled_from_networktables()` return False (omission 1).
- P5 Status/first-boot: with a real instance, `connection_count`/`topic_names` return without the error
  branch, and topic names include a published topic (C5).
- P6 `GetNetworktablesValue.run` returns a published double; returns None when the key is missing.
- P7 Compat guard: monkeypatch a mismatched wpimath shape and assert `ImportError`.
- Update the existing tests/fakes to the compat surface. Keep `test_position_contract` assertions via
  compat accessor helpers.

Java (2027 tree; the 2026 tree keeps its current suite):

- J1 Existing 10 contract tests ported to ns literals.
- J2 A 20 ms-old sample is accepted; a 600 ms-old and a +600 ms future sample are rejected. This is the
  audit's `TimingProbe` turned into a JUnit test.
- J3 `timestampSeconds` equals `Timer.getTimestamp()`-based expectation within 1 ms under HAL sim. The 2026
  example only compared against itself.
- J4 Warning throttling: two polls 1 ms apart produce a single warning. Inject a clock or count through a
  package-private hook. Interval assertion uses ns.
- J5 Sim: `EagleEyeCameraSim` with the camera exactly on a tag position does not throw (Optional bearing).
- Manual or nightly: the audit's `MixedJavaProbe` (2026 Python publisher → alpha Java server). Keep it as a
  documented script, since CI would need both environments.

## 5. Ownership splits (non-overlapping files)

| Agent | Owns | Deliverable |
|---|---|---|
| **Python** (Sol 6.1 medium) | `src/**` (incl. new `src/utils/wpilib_compat.py`, WebUI copy strings at `settings_tab_content.html:109` and `wizardState.js:169`), `tests/**`, `benchmarks/replay.py`, `pyproject.toml`, `uv.lock`, `install.sh`, `ci/build-pi-image.sh`, `.github/workflows/Pytests.yaml`, `pi-image.yml`, `dependabot.yml` | §3.1, §3.2, tests P1-P7; full pytest green in **both** extras; ruff/mypy clean |
| **Java** (Sol 6.1 medium) | `library/**` (SDK trees, both examples, parity tool, `library/README.md`, `library/VALIDATION.md`, Javadocs), `.github/workflows/JavaContract.yaml` | §3.3, tests J1-J5; 2026 build green with JDK17; 2027 Gradle build/test green with JDK25 |
| **Docs** (Sol 6.1 medium) | `EagleEye-Docs/docs/**`, root `README.md` of EagleEye-Vision-System | Versioned instructions (see below), after the Python/Java agents fix names/paths |

The docs agent should add a two-column **"WPILib 2026 / 2027 alpha-7"** treatment. Use Docusaurus `@theme/Tabs`, which is not currently used, so import it per page; alternatively, use clearly headed sections. Cover:

- `user-guide/robot-integration.md`: SDK paths, imports, `/1e6`→`/1e9` text at :236, FPGA wording, `Field`/`Fields` example at :280-293, and `Telemetry/EagleEye` vs `SmartDashboard/EagleEye` at :264-266. Add the warning semantics and the `setTimeSource` caveat.
- `user-guide/networktables.md`: a controller-agnostic address section. For Systemcore, state "use the controller's reachable address shown by …" only with a verified source (see U5).
- `user-guide/install.md` and `prepare-pi.md`: the `--wpilib` flag, and the glibc ≥ 2.39 / Debian 13 requirement for the alpha coprocessor stack. Make clear that a 2026 coprocessor works with alpha-7 robots.
- `codebase/data-systems/networktables.md:10-11,35` and `architecture/startup-sequence.md:37-38`: show the compat module, not raw calls.
- `pipelines/operations/secondary/publish-to-networktables.md`: the internal µs contract, the publish-boundary scaling, and the epoch per version.
- `codebase/testing.md`: the matrix legs, real-binding vs fake tests, P1-P5, and the parity script.

Merge order: Python and Java PRs are independent. Each opens against `main` from its own branch off
`feature/wpilib-2027-alpha7-support`, and the Docs PR (EagleEye-Docs) follows once names are final. Each PR body
states exactly which evidence was run, distinguishing loopback from hardware.

## 6. Unresolved choices (need a decision; defaults proposed)

- **U1 Java tree layout.** Default: keep `library/java/` = 2026 (zero churn for existing copy-paste users)
  and add `library/java-2027/`. Alternative: symmetric `library/java/wpilib-2026` + `wpilib-2027`.
  The alternative is cleaner but breaks every existing link and doc path.
- **U2 Python default.** Default: 2026 extra everywhere, with alpha opt-in only. Revisit when 2027 is stable
  and the Pi image moves to a glibc ≥ 2.39 base.
- **U3 Internal NT time unit.** Default: keep µs internally (§3.1). Migrating to ns would be churn with
  no wire-precision gain.
- **U4 Robot time base.** Default: `/1e9` direct, documented. Alternative: age-based
  `Timer.getTimestamp() - (now - ts)/1e9`, which survives `RobotController.setTimeSource`
  but changes the exact-equality contract test and the semantics for 2026 parity. Pick before the Java agent starts.
- **U5 Systemcore addressing.** The release notes require Systemcore image 14 / DS alpha 7, but this review did
  not verify Systemcore's default hostname/IP or USB address. Do not invent one. Either cite a WPILib 2027
  doc, or leave the generic "enter the controller's address" text. Optional: expose alpha `set_server_team`
  as a "team number" mode, a feature beyond migration, so it defaults to out of scope.
- **U6 Alert.** Default: `DriverStationErrors` only (parity). Optionally add an `org.wpilib.util.Alert`
  ("EagleEye topics unresolved", MEDIUM) set/cleared each poll. Verify it does not require a telemetry backend
  in the JUnit context before adopting it.
- **U7 Committing `reports/`.** The audit, probes and evidence contain absolute `/tmp` and `/Users/dark` paths.
  Proposed: commit `README.md` and this plan with the Java PR; keep probes/evidence uncommitted or move them
  to a release attachment.
- **U8 Pi base image pinning** (omission 13): pin Bookworm for the default image now, or move to Trixie (which would allow alpha installs)? Default: pin the current base and record it. Moving is a separate change.

## 7. Done criteria

- 2026 behavior unchanged: existing 39 Python tests (now among the full suite) and 10 Java tests pass. With
  the 2026 extra, Python publishes byte-identical payloads and timestamps.
- Alpha leg: full pytest passes with real `2027.0.0a7` bindings, including P1-P7. The 2027 Gradle project
  builds and tests under JDK 25, and the parity script passes.
- No remaining `getDefault|startClient4|getEntry|getInstance\(\)\.flush|isConnected|getConnections|getTopics|wpimath\.geometry|_now\(`
  outside `src/utils/wpilib_compat.py`. Enforce with a grep test.
- Docs build (`npm run build` in EagleEye-Docs) succeeds. No 2026 instruction removed without a versioned replacement.
- Physical Systemcore/DS/camera validation is explicitly **not** claimed. `VALIDATION.md` lists it as outstanding.

## 8. Adopted decisions and final results (October 13, 2026)

This appendix records the completed implementation. Sections 1–7 remain the historical
pre-implementation review; their proposed commands, ownership, and unresolved choices are
not authoritative descriptions of the final tree. The [completed support report](README.md)
supersedes the original unchanged-source compatibility verdict.

- **U1:** Adopted `library/java/` for 2026 and `library/java-2027/` for alpha-7,
  with separate runnable examples and an explicit SDK parity script.
- **U2:** Kept 2026 as the default Python environment. Implemented mutually exclusive
  **dependency groups**, rather than the proposed extras: default `wpilib2026`, opt-in
  `wpilib2027-alpha7`. Both groups share one lock; alpha pins are exact and 2026 is bounded.
  The installer persists `.wpilib-variant`, checks Linux glibc ≥ 2.39 for alpha, and startup
  reports/rejects incompatible runtime selections. CI exercises both groups.
- **U3:** Kept `TimingMetadata.capture_nt_us` in microseconds. The adapter normalizes
  alpha `_now()` by 1,000 and scales publisher times by 1,000. Startup checks clock rate,
  then caches the median epoch offset. Python timestamped samples expose `.time` in **both**
  releases; alpha's field uses nanoseconds. The proposed Java-style `.timestamp` spelling
  must not be applied to the Python binding.
- **U4:** Kept direct native-timestamp conversion (`/1e6` for 2026; `/1e9` for alpha).
  SDK/docs state the default robot time-source assumption and custom `setTimeSource` caveat.
- **U5:** Adopted controller-neutral address guidance and UI text. No guessed Systemcore
  address or team-number mode was introduced.
- **U6:** Adopted `DriverStationErrors.reportWarning` with the existing five-second
  throttle. No mandatory Alert or diagnostic stderr replacement was shipped.
- **U7:** Kept the publication-ready report and historical review. Before removal from
  the report tree, all original raw evidence/probes and documents were copied and
  checksum-verified in an ephemeral local archive. That archive is not a dependency
  of the checked-in regression suites.
- **U8:** Kept the image's existing configurable/latest base URL and explicitly selected
  the 2026 coprocessor. Build metadata records the actual OS/glibc/variant; a pinned base
  image and an executed Pi image build were not part of this validation.

Final real-binding Python validation: **411 passed, 8 skipped in each coherent environment**.
The runtime tests cover clock scale/epoch conversion, wrong-unit rejection, mixed/unsupported
bindings, pipeline disable/enable, missing values and numeric reads, monitoring/setup topic
APIs, and real client/server pose/meta timestamp equality with duplicate payloads.

Final Java validation: **13 tests and successful test/build for 2026**, **15 tests and
successful test/build for alpha-7**, plus a passing SDK parity check. Driver Station warning
and default clock semantics were tested through actual native/HAL artifacts. Frontend
validation passed **40 tests and the production build**; EagleEye-Docs production build passed.

The original 10-frame mixed-version Python/Java transport probes remain **historical audit
results**, distinct from final production regression runs. No physical controller/camera/DS
integration, Pi image build/flash/boot, or long-duration reconnection/clock/load validation is
claimed. A separate 2026.2.1 macOS native `readQueueStringArray` probe exposed an upstream
JNI constructor-signature crash; the passing 2026 pose/meta suite does not cover that failing
path. Alpha-7's detection-queue regression passed. See the support report and
`library/VALIDATION.md` for the limits of the software evidence.


## 9. Final review follow-up

The final independent Opus review found no blocking migration gaps. Its remaining clock-rate
check concern was resolved by replacing a single 5 ms window with up to three 50 ms attempts,
accepting any valid rate and rejecting only after all attempts fail. A deterministic test
covers an initial scheduler-skewed interval followed by a valid retry; wrong-unit rejection
remains covered. PubSubOptions keyword dispatch now uses a typed dictionary so either native
stub set can type-check the boundary. After these changes, both coherent Python environments
passed **411 tests with 8 skips**; scoped mypy validation covers compatibility and timing in
both versions. Other final validation and hardware limits remain as recorded above.

Startup logs also distinguish unmanaged developer environments from persisted installer
selections while preserving managed enforcement; a real-binding subprocess regression
verifies reported versions, and the subprocess code-string lint issue was fixed. Java CI
path filters were removed so root-level parity-affecting changes trigger contract checks,
and the parity script executable bit was corrected. The final rerun after all follow-ups
passed **411 tests with 8 skips in both coherent Python environments**. Scoped mypy checks
passed for both compatibility/timing modules in both environments.
