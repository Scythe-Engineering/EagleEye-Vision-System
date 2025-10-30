"""Simple YOLO Model Training Script for EagleEye Object Detection."""

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

    # Train with optional batch size
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "patience": patience,
        "project": "eagleeye_training",
        "name": run_name,
    }

    if batch_size:
        train_kwargs["batch"] = batch_size

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
