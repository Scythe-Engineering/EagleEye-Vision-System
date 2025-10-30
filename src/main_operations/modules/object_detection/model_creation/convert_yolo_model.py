"""Module for converting YOLO models to ONNX format."""

from ultralytics import YOLO


def load_and_convert_yolo_model() -> None:
    """Load a YOLO model from user input and convert it to ONNX format with opset 17."""
    yolo_model_path = input("Enter the path to the YOLO model file: ")
    yolo_model = YOLO(yolo_model_path)
    yolo_model.export(format="onnx")


if __name__ == "__main__":
    load_and_convert_yolo_model()
