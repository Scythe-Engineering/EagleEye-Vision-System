# Rust Implementations

This directory contains Rust extension modules for the EagleEye Object Detection system. These modules provide high-performance implementations of computationally intensive operations.

## Directory Structure

```
rust_implementations/
├── build.py                    # Master build script
├── modules/                    # All Rust modules
│   └── pose_outlier_filter/    # Example module
│       ├── Cargo.toml
│       ├── src/
│       │   └── lib.rs
│       └── test_integration.py
├── module_template/            # Template for new modules
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
└── README.md                   # This file
```

## Building Modules

All build commands use `uv run` to ensure proper dependency management.

### Build All Modules

```bash
uv run python build.py
```

This will automatically detect which modules have changed and need rebuilding.

### Force Rebuild All Modules

```bash
uv run python build.py --all
```

### Build Specific Module

```bash
uv run python build.py module_name
```

### List Available Modules

```bash
uv run python build.py --list
```

### Clean All Build Artifacts

```bash
uv run python build.py --clean
```

## Creating New Modules

### Using the Create Script (Recommended)

Use the provided script to create a new module from the template:

```bash
uv run python create_module.py module_name "Module description"
```

Example:

```bash
uv run python create_module.py image_processor "High-performance image processing functions"
```

This will automatically create the module structure and customize the template files.

### Manual Creation

1. Copy the template directory:

    ```bash
    cp -r module_template modules/your_new_module
    ```

2. Rename and customize the template files:

    - Replace `{{MODULE_NAME}}` with your actual module name
    - Replace `{{MODULE_CLASS_NAME}}` with your Python class name
    - Replace `{{MODULE_DESCRIPTION}}` with a description of your module

3. Implement your Rust code in `src/lib.rs`

4. Update `Cargo.toml` with your dependencies

5. Build the module:
    ```bash
    uv run python build.py your_new_module
    ```

## Module Requirements

Each module must have:

- `Cargo.toml` - Rust package configuration
- `src/lib.rs` - Main Rust source code

The master build.py script handles all building, dependency checking, and installation. It uses change detection based on file hashes to avoid unnecessary rebuilds.
