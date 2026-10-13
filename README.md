# EagleEye Vision System

EagleEye is a configurable robotics vision system for AprilTag localization and object detection. It runs multiple cameras and pipelines on Linux hardware, with a WebUI for setup and a robot-side NetworkTables library.

![EagleEye in action: live camera feed, dragging and connecting a pipeline operation, temporal acceleration, AprilTag detection previews, and an interactive 3D field view](media/eagleeye-demo.gif)

## Documentation

The [EagleEye documentation site](https://scythe-engineering.github.io/EagleEye-Docs/) contains installation, camera setup, pipeline configuration, robot integration, troubleshooting, developer documentation, and benchmark workflows.

- [Install EagleEye](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/install)
- [Build an AprilTag pipeline](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/pipeline-setup)
- [Add EagleEye to robot code](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/robot-integration)
- [Developer docs](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/overview)

## WPILib versions

The default Python coprocessor stack remains WPILib 2026 and communicates with 2027 alpha-7 robots over NT4. Use the robot SDK matching your project: [`library/java`](library/java) for 2026 or [`library/java-2027`](library/java-2027) for 2027 alpha-7. Runnable examples are in [`localization-sim`](library/examples/localization-sim) and [`localization-sim-2027`](library/examples/localization-sim-2027).

Alpha-7 Python is an explicit opt-in; Linux wheels require glibc ≥ 2.39. Debian 12 / Pi OS Bookworm keeps the default 2026 stack. See [installation variants](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/install#wpilib-coprocessor-variants), [versioned robot integration](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/robot-integration), and [SDK validation](library/VALIDATION.md) for commands and validation limits.

## Development

```bash
uv sync
uv run pytest -q tests
npm run build
```

See the [testing guide](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/testing) for test markers and hardware-specific checks.

## License

EagleEye Vision System is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). See [`LICENSE`](LICENSE) for the full terms.
