# EagleEye Vision System

Hello! EagleEye is a dynamic vision system for robotics, built for teams who want fast AprilTag localization and object detection without giving up control over how it all works. New to vision? Start with the setup wizard and ready-made pipelines. Want to get deep into it? Open the flow graph and make it your own.

The goal of this project is simple: make advanced robot vision easy to start with, fast on affordable hardware, and easy to extend when your robot needs something different.

> **THIS PROJECT IS IN ALPHA!!!! Expect issues and problems.** I've done a ton of simulation and testing, but I am only one person. Try it, break it, and please tell me what went wrong.

**[Setup and installation](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/install) · [Using EagleEye](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/overview) · [Robot integration](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/robot-integration)**

![EagleEye in action: live camera feed, dragging and connecting a pipeline operation, temporal acceleration, AprilTag detection previews, and an interactive 3D field view](docs/media/eagleeye-demo.gif)

## Why EagleEye?

- Run multiple cameras and inference pipelines natively on one device. Run AprilTag detection and object detection on the same camera at the same time, without switching between them.
- Build new functionality in the visual flow graph. Connect detection, filtering, pose estimation, and robot outputs, or start with a bundled template and change only what you need.
- Get live camera views for drivers, inspect what individual operations see, and check your robot's estimated pose in the 3D field view. Calibration lives in the browser too.
- Use the MemryX MX3 AI accelerator for native, asynchronous object detection with low latency and high throughput. The accelerator is around US$150, separate from the computer and cameras.
- Send results to your robot through NetworkTables and the included robot-side library. Keep the vision plumbing out of your drivetrain code.
- Extend the Python backend with your own operations. Write them in Python, or use Rust for the performance-heavy parts. You don't need to rewrite the system to add one new idea.

## Some actual numbers

These are my test results so it may not repersent exactly what you will get.

| Workload | Hardware | Reported performance |
| --- | --- | --- |
| AprilTag detection, one camera | Raspberry Pi 5, 8 GB | **120 FPS** at native 1280 × 800* |
| AprilTag detection, three cameras at once | Raspberry Pi 5, 8 GB | **60 FPS per camera** at native 1280 × 800* |
| YOLO26 small object detection | MemryX MX3 accelerator | **60+ FPS** |

\* The AprilTag results use [temporal acceleration](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/temporal-acceleration). It predicts where tags will appear and searches those regions at native image resolution instead of scanning the entire frame. FPS varies with field position and the regions being searched. In my testing, it has consistently been faster than full-frame detection. These are AprilTag-only results, not measurements with object detection running alongside it.

## How it compares

The main stand-out functionaltiy of EagleEye are the editable flow graph, simultaneous detection types, and multi-camera performance on a single Pi. Also the frontend actually looks good (though may have ass code)

| Feature | EagleEye | Limelight | PhotonVision |
| --- | :---: | :---: | :---: |
| AprilTag localization | ✅ | ✅ | ✅ |
| Web configuration, live views, and NetworkTables | ✅ | ✅ | ✅ |
| Neural object detection on supported hardware | ✅ | ✅ | ✅ |
| Multiple cameras processing on one host | ✅ | ![Yellow check](https://img.shields.io/badge/-%E2%9C%93-yellow) | ✅ |
| AprilTags + neural detection on the same camera at once | ✅ | ❌ | ❌ |
| Visual flow graph editor | ✅ | ❌ | ❌ |
| Runs on third-party Linux hardware | ✅ | ❌ | ✅ |

And you can easely extend it to add more functionality!

## Give it a try

Install with a Raspberry Pi image or the one-command installer, then let the first-boot wizard walk you through camera setup. EagleEye targets Linux on most devices, but Raspberry Pis are the most tested hardware. Orange Pis are strange and may not work perfectly. Other Linux hardware may need extra setup.

The **[docs website](https://scythe-engineering.github.io/EagleEye-Docs/)** handles installation, calibration, pipeline setup, robot integration, and troubleshooting. All the instructions live there.

Ui EX:
![Live camera view in the EagleEye browser interface](https://scythe-engineering.github.io/EagleEye-Docs/img/ui-screenshots/views-tab.png)

## Contributing

Contributions are very welcome. Bug reports, testing on actual robots, and docs fixes help just as much as new features.

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request explaining what changed and how you tested it.
4. We'll review it and work through anything that needs fixing.

Start with the [developer docs](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/overview), including [tests and pre-push checks](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/testing#enable-the-pre-push-checks).

## License

EagleEye Vision System © 2025–2026 ScytheEngineering, licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
See [LICENSE](LICENSE) for the full terms.

Every FIRST team, student, school, and hobbyist can use, modify, and share EagleEye for free for noncommercial purposes under those terms. Commercial use requires a separate commercial license. Contact darkedenc9@gmail.com first.

## Contributors

- [DarkEden-coding](https://github.com/DarkEden-coding) - Main contributor

![Repository activity and contribution statistics](https://repobeats.axiom.co/api/embed/afdf811c96a1e587ab15608b17e83b7880631ffc.svg "Repobeats analytics image")
