#!/usr/bin/env python3
"""Check SDK behavioral parity, allowing only the documented WPILib API changes."""

from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[1]


def migrate(source: str) -> str:
    """Apply the documented 2026-to-2027 Java SDK substitutions.

    Replacements run in insertion order, so Matrix and VecBuilder must precede
    the broader math package prefix replacement. The fully qualified
    AprilTagFieldLayout replacement must precede the simple class name.

    Args:
        source: Java SDK source written against WPILib 2026.

    Returns:
        Source normalized to the supported 2027 alpha-7 API.
    """
    replacements = {
        "edu.wpi.first.math.Matrix": "org.wpilib.math.linalg.Matrix",
        "edu.wpi.first.math.VecBuilder": "org.wpilib.math.linalg.VecBuilder",
        "edu.wpi.first.math.": "org.wpilib.math.",
        "edu.wpi.first.networktables.": "org.wpilib.networktables.",
        "edu.wpi.first.wpilibj.DriverStation": "org.wpilib.driverstation.DriverStationErrors",
        "DriverStation.reportWarning": "DriverStationErrors.reportWarning",
        "edu.wpi.first.apriltag.AprilTagFieldLayout": "org.wpilib.fields.Field",
        "AprilTagFieldLayout": "Field",
        "AprilTagFields.kDefaultField": "Fields.DEFAULT_FIELD",
        "PubSubOption.sendAll(true)": "PubSubOption.SEND_ALL",
        "PubSubOption.keepDuplicates(true)": "PubSubOption.KEEP_DUPLICATES",
        "NT_TIME_UNITS_PER_SECOND = 1e6": "NT_TIME_UNITS_PER_SECOND = 1e9",
        "Micros": "Nanos",
        "latestDetectionsTimestamp": "latestDetectionsTimestampNanos",
        "tag.pose": "tag.getPose()",
        "long now =": "long nowNanos =",
        ", now)": ", nowNanos)",
        "long timestamp)": "long timestampNanos)",
        "== timestamp)": "== timestampNanos)",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        "      Rotation2d bearing = offset.getAngle().minus(cameraHeading);",
        "      var angle = offset.getAngle();\n"
        "      if (angle.isEmpty()) {\n"
        "        // No planar bearing when the camera is directly above or below the tag.\n"
        "        continue;\n"
        "      }\n"
        "      Rotation2d bearing = angle.get().minus(cameraHeading);",
    )
    source = source.replace(
        " * translates that into roboRIO FPGA time on arrival. The timestamp on a received sample is\n"
        " * therefore already in the domain {@code addVisionMeasurement} expects: do not subtract a latency,\n"
        " * and do not use {@code Timer.getFPGATimestamp()} as the measurement time.",
        " * translates that into NT/robot monotonic time on arrival (simulated time in HAL simulation).\n"
        " * Dividing its nanosecond timestamp by 1e9 gives the default {@code Timer.getTimestamp()}\n"
        " * time base expected by {@code addVisionMeasurement}; do not subtract latency or use the\n"
        " * current time as measurement time. A custom {@code RobotController.setTimeSource} must\n"
        " * preserve that time base, or robot code must translate observations before fusion.",
    )
    return source


def main() -> int:
    """Compare the 2027 Java SDK against normalized 2026 sources.

    Returns:
        0 if both source files match, or 1 if any differences are found.
    """
    failed = False
    for filename in ("EagleEyeCamera.java", "EagleEyeCameraSim.java"):
        expected = migrate((ROOT / "java/frc/robot/vision" / filename).read_text())
        actual = (ROOT / "java-2027/frc/robot/vision" / filename).read_text()
        if expected != actual:
            failed = True
            print(
                "".join(
                    difflib.unified_diff(
                        expected.splitlines(True),
                        actual.splitlines(True),
                        fromfile="2026 normalized/" + filename,
                        tofile="2027/" + filename,
                    )
                )
            )
    if not failed:
        print("Java SDK parity verified (2026 / 2027 alpha-7).")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
