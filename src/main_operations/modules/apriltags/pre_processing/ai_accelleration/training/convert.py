# mostly not needed due to direct output of onnx models from training

import os
import argparse
import torch
import torch.onnx

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.grid_detectors.predictor import GridPredictor
from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import TARGET_WIDTH, TARGET_HEIGHT, MODEL_PATH


def load_pytorch_model(model_path: str) -> GridPredictor:
    """Load the trained PyTorch model from file.
    
    Args:
        model_path (str): Path to the saved PyTorch model (.pth file).
        
    Returns:
        GridPredictor: Loaded model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GridPredictor()
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model.to(device)


def create_dummy_input(batch_size: int = 1) -> torch.Tensor:
    """Create dummy input tensor for ONNX export.
    
    Args:
        batch_size (int): Batch size for the dummy input.
        
    Returns:
        torch.Tensor: Dummy input tensor of shape (batch_size, 1, TARGET_HEIGHT, TARGET_WIDTH).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn(batch_size, 1, TARGET_HEIGHT, TARGET_WIDTH, device=device)


def convert_to_onnx(
    model: GridPredictor, 
    dummy_input: torch.Tensor, 
    output_path: str,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None
) -> None:
    """Convert PyTorch model to ONNX format.
    
    Args:
        model (GridPredictor): The PyTorch model to convert.
        dummy_input (torch.Tensor): Dummy input tensor for tracing.
        output_path (str): Path where the ONNX model will be saved.
        input_names (list[str], optional): Names for input tensors.
        output_names (list[str], optional): Names for output tensors.
        dynamic_axes (dict, optional): Dynamic axes specification for variable input sizes.
    """
    if input_names is None:
        input_names = ["input_image"]
    
    if output_names is None:
        output_names = ["grid_predictions"]
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input_image": {0: "batch_size"},
            "grid_predictions": {0: "batch_size"}
        }
    
    print("Converting model to ONNX format...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model successfully converted to ONNX: {output_path}")


def verify_onnx_model(onnx_path: str, pytorch_model: GridPredictor, dummy_input: torch.Tensor) -> bool:
    """Verify that the ONNX model produces the same output as the PyTorch model.
    
    Args:
        onnx_path (str): Path to the ONNX model.
        pytorch_model (GridPredictor): Original PyTorch model.
        dummy_input (torch.Tensor): Input tensor for comparison.
        
    Returns:
        bool: True if outputs match within tolerance, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not available. Skipping verification.")
        return True
    
    print("Verifying ONNX model output...")
    
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()
    
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {"input_image": dummy_input.cpu().numpy()})[0]
    
    max_diff = abs(pytorch_output - onnx_output).max()
    tolerance = 2e-5
    
    if max_diff < tolerance:
        print(f"✓ Verification passed! Max difference: {max_diff:.2e}")
        return True
    else:
        print(f"✗ Verification failed! Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
        return False


def main() -> None:
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument(
        "--input-model", 
        type=str, 
        default=MODEL_PATH,
        help=f"Path to input PyTorch model (.pth file). Default: {MODEL_PATH}"
    )
    parser.add_argument(
        "--output-model", 
        type=str, 
        help="Path for output ONNX model. Default: same directory as input with .onnx extension"
    )
    parser.add_argument(
        "--skip-verification", 
        action="store_true",
        help="Skip verification step (faster but less safe)"
    )
    
    args = parser.parse_args()
    
    model_path = args.input_model
    if args.output_model:
        onnx_output_path = args.output_model
    else:
        onnx_output_path = os.path.join(os.path.dirname(model_path), "model.onnx")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading PyTorch model from: {model_path}")
    model = load_pytorch_model(model_path)
    
    print(f"Creating dummy input with shape: (1, 1, {TARGET_HEIGHT}, {TARGET_WIDTH})")
    dummy_input = create_dummy_input(batch_size=1)
    
    convert_to_onnx(model, dummy_input, onnx_output_path)
    
    if not args.skip_verification:
        verify_onnx_model(onnx_output_path, model, dummy_input)
    else:
        print("Skipping verification as requested.")
    
    print("\nConversion complete!")
    print(f"Original model: {model_path}")
    print(f"ONNX model: {onnx_output_path}")
    print(f"Model input shape: (batch_size, 1, {TARGET_HEIGHT}, {TARGET_WIDTH})")
    print("Model output shape: (batch_size, 20, 20)")


if __name__ == "__main__":
    main()
