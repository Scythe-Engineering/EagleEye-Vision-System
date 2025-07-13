from ultralytics import YOLO


def select_format():
    print("Select the format to export the model:")
    formats = {
        "1": "PyTorch (Best for CPU and GPU-based inference)",
        "2": "TorchScript (Optimized for deployment)",
        "3": "ONNX (Cross-platform support)",
        "4": "OpenVINO (Best for Intel hardware acceleration)",
        "5": "TensorRT (Best for NVIDIA GPUs)",
        "6": "CoreML (Best for Apple devices)",
        "7": "TF SavedModel (For TensorFlow environments)",
        "8": "TF GraphDef (Older TensorFlow format)",
        "9": "TF Lite (Optimized for mobile and embedded devices)",
        "10": "TF Edge TPU (Best for Edge TPU accelerators, like Coral)",
        "11": "TF.js (For use in web environments)",
        "12": "PaddlePaddle (For PaddlePaddle frameworks)",
        "13": "MNN (For mobile and embedded systems)",
        "14": "NCNN (For ARM and mobile devices)",
        "15": "IMX500 (For i.MX500 hardware)",
        "16": "RKNN (For Rockchip devices)",
    }

    for key, value in formats.items():
        print(f"{key}: {value}")

    choice = input("Enter your choice (1-16): ")

    format_dict = {
        "1": "pytorch",
        "2": "torchscript",
        "3": "onnx",
        "4": "openvino",
        "5": "engine",
        "6": "coreml",
        "7": "saved_model",
        "8": "pb",
        "9": "tflite",
        "10": "edgetpu",
        "11": "tfjs",
        "12": "paddle",
        "13": "mnn",
        "14": "ncnn",
        "15": "imx",
        "16": "rknn",
    }

    return format_dict.get(choice, None)


def export_model():
    # Load YOLO model
    model = YOLO(model_path)

    # Export to selected format
    export_path = model.export(format=selected_format)

    print(f"Model exported to {export_path}")
    return export_path


if __name__ == "__main__":
    model_path = input("Enter the model path: ")

    # Select format
    selected_format = select_format()
    if selected_format:
        export_model()
    else:
        print("Invalid format selected!")
