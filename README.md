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
performs fresh installs only. If the target directory already exists, it refuses
and points you at **Settings -> System Update** in the Web UI
(`http://<pi-address>:5001`).

### Raspberry Pi image

Tagged releases include an `.rpi-imager-manifest` asset. Open that file with the
latest Raspberry Pi Imager, select EagleEye Vision System, and enter the Wi-Fi
network or phone hotspot in OS customization before writing the card. The image
starts SSH with username `eagleeye` and password `eagleeye`; change the password
after the first login. On models with USB OTG support, USB gadget networking is
enabled on first boot. Connect the OTG port directly to the computer and use
`ssh eagleeye@10.12.194.1` or `ssh eagleeye@eagleeye.local`. On Pi 3/3B+, connect
over the configured Wi-Fi or Ethernet network instead. The first boot may restart
once while it enables the gadget.

A fresh install opens the first-boot wizard. It configures each camera and
creates the selected localization and detection pipelines after calibration.

For further instructions, please refer to the [wiki page](https://github.com/frc3322/EagleEye-Object-Detection/wiki).

## Robot library

Robot-side code for consuming EagleEye lives in [`library/`](library/), with full usage notes in
[`library/README.md`](library/README.md).

Copy `library/java/frc/robot/vision/EagleEyeCamera.java` into your robot project, create a pipeline
from the WebUI's "Basic localization" template, and feed your pose estimator:

```java
public class Drive extends SubsystemBase {
  // Basic localization publishes pose and meta; forSource would also expect detections.
  private final EagleEyeCamera[] cameras = {
    new EagleEyeCamera("localization/front/pose", "localization/front/meta")
  };

  @Override
  public void periodic() {
    poseEstimator.update(gyro.getRotation2d(), modulePositions);
    EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
  }
}
```

Robot code supplies the NetworkTables keys, so any pipeline layout works. Pipelines from a
detection template also publish `localization/<source>/detections`; for those sources build the
camera with `EagleEyeCamera.forSource("localization/front")`, which subscribes to all three topics.
Keys must match the `target_key` set in the WebUI; one that nothing publishes raises a Driver
Station warning naming the key instead of failing silently.

Each localization source publishes two NetworkTables topics that share one capture timestamp:

| Topic | Type | Contents |
| --- | --- | --- |
| `EagleEye/localization/<source>/pose` | `Pose3d` struct | Robot pose in field coordinates |
| `EagleEye/localization/<source>/meta` | `double[3]` | `[tagCount, meanTagDistanceMeters, reprojectionErrorPixels]` |
| `EagleEye/localization/<source>/detections` | `String[]` | Repeating `[className, fieldX, fieldY, fieldZ]` groups (detection templates only) |

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
EagleEye Vision System © 2025-2026 ScytheEngineering, licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
See the [LICENSE](LICENSE) file for the full terms.

In plain language: every FIRST team, student, school, and hobbyist can use, modify, and
share EagleEye for free, forever — noncommercial use is unrestricted. Commercial use
requires a separate commercial license: contact darkedenc9@gmail.com first.
<img width="1606" height="979" alt="image" src="https://github.com/user-attachments/assets/00b03576-f924-415a-a8c5-8559e4f1a509" />



## Contributors
- [DarkEden-coding](https://github.com/DarkEden-coding) - Main Contributor
