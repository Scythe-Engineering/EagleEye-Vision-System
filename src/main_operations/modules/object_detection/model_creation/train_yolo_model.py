"""YOLO Model Training Script with Data Augmentation for EagleEye Object Detection."""

import os

from ultralytics import YOLO

# Try to import torch for GPU detection
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def main() -> None:
    """Simple YOLO training with user input."""
    print("=== EagleEye YOLO Model Training ===")

    # Check CUDA/GPU availability
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ GPU available: {gpu_count} device(s)")
            print(f"   Using: {device_name}")
        else:
            print("❌ GPU not available - training will use CPU (slower)")
    else:
        print("⚠️  PyTorch not available - cannot check GPU status")
        print("   Training device will be auto-detected by Ultralytics")

    # Get user inputs with validation
    data_dir = input("Enter data directory path (containing data.yaml): ").strip()

    try:
        epochs = int(input("Enter number of epochs: ").strip())
    except ValueError:
        print("Error: Epochs must be a number!")
        return

    try:
        patience = int(input("Enter patience (early stopping): ").strip())
    except ValueError:
        print("Error: Patience must be a number!")
        return

    model_name = input(
        "Enter starting model (yolov8n, yolov8s, yolov8m, etc.) or path to trained model (.pt file): "
    ).strip()

    # Optional: custom batch size
    batch_size_input = input("Enter batch size (press Enter for auto): ").strip()
    batch_size = None
    if batch_size_input:
        try:
            batch_size = int(batch_size_input)
        except ValueError:
            print("Error: Batch size must be a number!")
            return

    # Optional: custom image size
    img_size_input = input(
        "Enter image size (320, 640, or press Enter for auto): "
    ).strip()
    img_size = None
    if img_size_input:
        try:
            img_size = int(img_size_input)
            if img_size not in [320, 640]:
                print(
                    "Warning: Unusual image size selected. Common sizes are 320 or 640."
                )
        except ValueError:
            print("Error: Image size must be a number!")
            return

    # Optional: augmentation settings
    print("\n--- Augmentation Settings ---")
    augmentation_input = (
        input("Enable data augmentation? (y/n, press Enter for auto): ").strip().lower()
    )
    enable_augmentation = augmentation_input == "y"

    if enable_augmentation:
        # HSV hue augmentation
        hsv_h_input = input(
            "HSV hue augmentation (0.0-0.5, press Enter for 0.015): "
        ).strip()
        hsv_h = 0.015 if not hsv_h_input else float(hsv_h_input)

        # Geometric augmentations
        degrees_input = input(
            "Rotation degrees (-180 to 180, press Enter for 0.0): "
        ).strip()
        degrees = 0.0 if not degrees_input else float(degrees_input)

        translate_input = input(
            "Translation factor (0.0-1.0, press Enter for 0.1): "
        ).strip()
        translate = 0.1 if not translate_input else float(translate_input)

        scale_input = input("Scale factor (0.0-1.0, press Enter for 0.5): ").strip()
        scale = 0.5 if not scale_input else float(scale_input)

        shear_input = input(
            "Shear degrees (-180 to 180, press Enter for 0.0): "
        ).strip()
        shear = 0.0 if not shear_input else float(shear_input)

        perspective_input = input(
            "Perspective factor (0.0-0.001, press Enter for 0.0): "
        ).strip()
        perspective = 0.0 if not perspective_input else float(perspective_input)
    else:
        # Default augmentation settings (minimal)
        hsv_h = 0.015
        degrees = 0.0
        translate = 0.1
        scale = 0.5
        shear = 0.0
        perspective = 0.0

    # Validate inputs
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return

    data_yaml = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found in '{data_dir}'!")
        return

    print("\nStarting training with:")
    print(f"  Data: {data_yaml}")
    if os.path.exists(model_name):
        print(f"  Model: {os.path.basename(model_name)} (custom trained)")
    else:
        print(f"  Model: {model_name} (pretrained)")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}")
    if batch_size:
        print(f"  Batch size: {batch_size}")
    if img_size:
        print(f"  Image size: {img_size}x{img_size}")

    print("\n  Augmentation:")
    print(f"    HSV Hue: {hsv_h}")
    print(f"    Rotation: {degrees}°")
    print(f"    Translation: {translate}")
    print(f"    Scale: {scale}")
    print(f"    Shear: {shear}°")
    print(f"    Perspective: {perspective}")

    # Load model
    model = YOLO(model_name)

    # Generate training run name
    if os.path.exists(model_name):
        # For custom models, use the filename without extension
        base_name = os.path.splitext(os.path.basename(model_name))[0]
        run_name = f"{base_name}_continued"
    else:
        # For standard models, use the model name
        run_name = f"{model_name}_train"

    # Train with optional parameters
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "patience": patience,
        "project": "eagleeye_training",
        "name": run_name,
        # Augmentation parameters
        "hsv_h": hsv_h,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
    }

    if batch_size:
        train_kwargs["batch"] = batch_size

    if img_size:
        train_kwargs["imgsz"] = img_size

    model.train(**train_kwargs)

    print("\nTraining completed successfully!")

    # Optional: Run validation
    validate_input = input("\nRun validation on trained model? (y/n): ").strip().lower()
    if validate_input == "y":
        print("Running validation...")
        model.val(data=data_yaml)
        print("Validation completed!")


if __name__ == "__main__":
    main()
