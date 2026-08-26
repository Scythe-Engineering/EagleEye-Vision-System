[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) [![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) [![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) 

# EagleEye-Vision-System

**EagleEye Vision System** is a Python-based project aimed at detecting game pieces using YOLO Ai object detection. This project was created by DarkEden-Coding on the FRC Team 3322 Eagle Evolution.

![Alt](https://repobeats.axiom.co/api/embed/afdf811c96a1e587ab15608b17e83b7880631ffc.svg "Repobeats analytics image")

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Robot library](#robot-library)
- [Contributing](#contributing)
- [License](#license)

## Introduction
EagleEye Object Detection is designed to help identify and track game pieces in 3d space over different years. Code similarity between years is a priority where teams can simply create their detection model for each year while keeping the same simple and reliable code.

## Installation

Run this as the normal (non-root) sudo-capable user that should own the install:

```bash
(
  installer="$(mktemp)" &&
  trap 'rm -f "$installer"' EXIT &&
  curl -fsSL https://raw.githubusercontent.com/Scythe-Engineering/EagleEye-Vision-System/main/install.sh -o "$installer" &&
  bash "$installer"
)
```

The installer clones into `~/EagleEye-Vision-System`, installs the apt packages,
uv, Node.js, and Rust, syncs the Python dependencies (including MemryX) and the
frontend build, adds the user to the camera device groups, then installs,
enables, and starts the `eagleeye` systemd service. It is tested on Raspberry Pi
OS Lite 64-bit (Debian 12, arm64); other platforms warn and continue. It
performs fresh installs only — if the target directory already exists it refuses
and points you at **Settings -> System Update** in the Web UI
(`http://<pi-address>:5001`).

A fresh install ships an intentionally incomplete `2026_apriltag_starter`
pipeline (Device Input -> Detect AprilTags -> PnP Camera Localization ->
Camera to Robot Pose -> Robot Pose Output). Fill in the camera bus ID,
calibration, extrinsics, and 2026 AprilTag map path in the Web UI before it will
run; until then it stays inactive.

For further instructions, please refer to the [wiki page](https://github.com/frc3322/EagleEye-Object-Detection/wiki).

## Robot library

Robot-side code for consuming EagleEye lives in [`library/`](library/), with full usage notes in
[`library/README.md`](library/README.md).

Copy `library/java/frc/robot/vision/EagleEyeCamera.java` into your robot project, create a pipeline
from the WebUI's "Basic localization" template, and feed your pose estimator:

```java
public class Drive extends SubsystemBase {
  private final EagleEyeCamera[] cameras = {EagleEyeCamera.forSource("localization/front")};

  @Override
  public void periodic() {
    poseEstimator.update(gyro.getRotation2d(), modulePositions);
    EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
  }
}
```

Robot code supplies the NetworkTables keys, so any pipeline layout works — pass both keys
explicitly with `new EagleEyeCamera(poseKey, metaKey)` when they do not follow the preset. Keys
must match the `target_key` set in the WebUI; one that nothing publishes raises a Driver Station
warning naming the key instead of failing silently.

Each localization source publishes two NetworkTables topics that share one capture timestamp:

| Topic | Type | Contents |
| --- | --- | --- |
| `EagleEye/localization/<source>/pose` | `Pose3d` struct | Robot pose in field coordinates |
| `EagleEye/localization/<source>/meta` | `double[3]` | `[tagCount, meanTagDistanceMeters, reprojectionErrorPixels]` |

Poses are stamped with the camera's kernel-level exposure timestamp and translated into roboRIO
FPGA time by ntcore, so robot code passes the received timestamp straight to
`addVisionMeasurement`. There is no latency to subtract. The metrics topic is what the library
derives its standard deviations from.

## Contributing
We welcome contributions to improve EagleEye Object Detection. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.
4. We will review your changes and merge them if they meet our standards

You can see the status of the current code in terms of cleanliness and issues at this link: [sonar cloud](https://sonarcloud.io/project/overview?id=Scythe-Engineering_EagleEye-Object-Detection)

### Enable the pre-push checks

After cloning, enable the repository's tracked Git hooks once:

```bash
git config core.hooksPath .githooks
```

The pre-push hook runs the same dependency sync, WebUI production build, and
full pytest suite used by GitHub Actions. Git cancels the push if any check
fails. As with all local Git hooks, `git push --no-verify` explicitly bypasses
this check; protected-branch CI should remain required as the authoritative
gate.

## License
EagleEye Framework © 2025 by ScytheEngineering is licensed under CC BY-NC 4.0. See the [LICENSE](LICENSE) file for details.
<img width="1606" height="979" alt="image" src="https://github.com/user-attachments/assets/00b03576-f924-415a-a8c5-8559e4f1a509" />



## Contributors
- [DarkEden-coding](https://github.com/DarkEden-coding) - Main Contributor
