import cv2
import numpy as np
from pathlib import Path
import time
from typing import Tuple, Optional, List

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array, or None if loading failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    except Exception as error:
        print(f"Error loading image: {error}")
        return None

def squash_colors_to_targets(
    image: np.ndarray, 
    target_color_1: Tuple[int, int, int], 
    target_color_2: Tuple[int, int, int],
    color_tolerance: int = 50
) -> np.ndarray:
    """Squash image colors to the closest target color within tolerance, replace others with pink.
    
    Args:
        image: Input image as numpy array (BGR format)
        target_color_1: First target color as (B, G, R) tuple
        target_color_2: Second target color as (B, G, R) tuple
        color_tolerance: Maximum distance from target colors to squash pixels
        
    Returns:
        Squashed image with pixels set to exact target colors or pink
    """
    print(f"Squashing colors with tolerance: {color_tolerance}")
    print(f"Target color 1 (BGR): {target_color_1}")
    print(f"Target color 2 (BGR): {target_color_2}")
    
    pink_color = (147, 20, 255)  # Pink in BGR format
    squashed_image = np.full_like(image, pink_color)
    
    target_1_bgr = np.array(target_color_1, dtype=np.float32)
    target_2_bgr = np.array(target_color_2, dtype=np.float32)
    
    image_float = image.astype(np.float32)
    
    distance_to_target_1 = np.sqrt(np.sum((image_float - target_1_bgr) ** 2, axis=2))
    distance_to_target_2 = np.sqrt(np.sum((image_float - target_2_bgr) ** 2, axis=2))
    
    mask_closer_to_target_1 = (distance_to_target_1 <= distance_to_target_2) & (distance_to_target_1 <= color_tolerance)
    mask_closer_to_target_2 = (distance_to_target_2 < distance_to_target_1) & (distance_to_target_2 <= color_tolerance)
    
    squashed_image[mask_closer_to_target_1] = target_color_1
    squashed_image[mask_closer_to_target_2] = target_color_2
    
    pixels_squashed_to_target_1 = np.sum(mask_closer_to_target_1)
    pixels_squashed_to_target_2 = np.sum(mask_closer_to_target_2)
    total_squashed_pixels = pixels_squashed_to_target_1 + pixels_squashed_to_target_2
    replaced_pixels = np.sum(~(mask_closer_to_target_1 | mask_closer_to_target_2))
    total_pixels = image.shape[0] * image.shape[1]
    
    squash_percentage = (total_squashed_pixels / total_pixels) * 100
    replacement_percentage = (replaced_pixels / total_pixels) * 100
    
    print(f"Squashed {pixels_squashed_to_target_1:,} pixels to target color 1")
    print(f"Squashed {pixels_squashed_to_target_2:,} pixels to target color 2")
    print(f"Total squashed pixels: {total_squashed_pixels:,} ({squash_percentage:.2f}%)")
    print(f"Replaced {replaced_pixels:,} pixels ({replacement_percentage:.2f}%) with pink")
    
    return squashed_image

def filter_colors_by_target_colors(
    image: np.ndarray, 
    target_color_1: Tuple[int, int, int], 
    target_color_2: Tuple[int, int, int],
    color_tolerance: int = 50
) -> np.ndarray:
    """Filter image to keep only pixels close to two target colors, replace others with pink.
    
    Args:
        image: Input image as numpy array (BGR format)
        target_color_1: First target color as (B, G, R) tuple
        target_color_2: Second target color as (B, G, R) tuple
        color_tolerance: Maximum distance from target colors to keep pixels
        
    Returns:
        Filtered image with target colors preserved and other pixels replaced with pink
    """
    print(f"Filtering colors with tolerance: {color_tolerance}")
    print(f"Target color 1 (BGR): {target_color_1}")
    print(f"Target color 2 (BGR): {target_color_2}")
    
    pink_color = (147, 20, 255)  # Pink in BGR format
    filtered_image = np.full_like(image, pink_color)
    
    target_1_bgr = np.array(target_color_1, dtype=np.float32)
    target_2_bgr = np.array(target_color_2, dtype=np.float32)
    
    image_float = image.astype(np.float32)
    
    distance_to_target_1 = np.sqrt(np.sum((image_float - target_1_bgr) ** 2, axis=2))
    distance_to_target_2 = np.sqrt(np.sum((image_float - target_2_bgr) ** 2, axis=2))
    
    mask_color_1 = distance_to_target_1 <= color_tolerance
    mask_color_2 = distance_to_target_2 <= color_tolerance
    combined_mask = mask_color_1 | mask_color_2
    
    filtered_image[combined_mask] = image[combined_mask]
    
    preserved_pixels = np.sum(combined_mask)
    replaced_pixels = np.sum(~combined_mask)
    total_pixels = image.shape[0] * image.shape[1]
    preservation_percentage = (preserved_pixels / total_pixels) * 100
    replacement_percentage = (replaced_pixels / total_pixels) * 100
    
    print(f"Preserved {preserved_pixels:,} pixels ({preservation_percentage:.2f}%) out of {total_pixels:,}")
    print(f"Replaced {replaced_pixels:,} pixels ({replacement_percentage:.2f}%) with pink")
    
    return filtered_image

def apply_canny_edge_detection(image: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection to an image with very high sensitivity.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Edge-detected image as numpy array
    """
    low_threshold = 10
    high_threshold = 50
    blur_kernel_size = 1
    
    print("Using very high sensitivity:")
    print(f"  Low threshold: {low_threshold}")
    print(f"  High threshold: {high_threshold}")
    print(f"  Blur kernel size: {blur_kernel_size}")
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel_size > 1:
        blurred_image = cv2.GaussianBlur(grayscale_image, (blur_kernel_size, blur_kernel_size), 0)
    else:
        blurred_image = grayscale_image
    
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges

def detect_straight_lines(edge_image: np.ndarray) -> Tuple[np.ndarray, List]:
    """Detect straight lines using Line Segment Detector (LSD) algorithm.
    
    Args:
        edge_image: Edge-detected image as numpy array
        
    Returns:
        Tuple of (lines_image, detected_lines_list)
    """
    lines_image = np.zeros_like(edge_image)
    
    scale = 0.8
    sigma_scale = 0.6
    quant = 2.0
    angle_tolerance = 22.5
    log_eps = 0.0
    density_threshold = 0.5
    n_bins = 1024
    minimum_line_size = 60
    
    print("Detecting straight lines using Line Segment Detector (LSD):")
    print(f"  Scale: {scale}")
    print(f"  Sigma scale: {sigma_scale}")
    print(f"  Quantization error bound: {quant}")
    print(f"  Angle tolerance: {angle_tolerance}")
    print(f"  Minimum line size filter: {minimum_line_size}")
    
    line_segment_detector = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD,
        scale=scale,
        sigma_scale=sigma_scale,
        quant=quant,
        ang_th=angle_tolerance,
        log_eps=log_eps,
        density_th=density_threshold,
        n_bins=n_bins
    )
    
    detected_lines, _, _, _ = line_segment_detector.detect(edge_image)
    
    if detected_lines is not None and len(detected_lines) > 0:
        print(f"Detected {len(detected_lines)} straight lines before filtering")
        
        filtered_lines = []
        for i, line in enumerate(detected_lines):
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if line_length >= minimum_line_size:
                filtered_line = [[int(x1), int(y1), int(x2), int(y2)]]
                filtered_lines.append(filtered_line)
                cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,), 2)
        
        print(f"Kept {len(filtered_lines)} lines after minimum size filtering")
        return lines_image, filtered_lines
    else:
        print("No straight lines detected")
        return lines_image, []

def create_comparison_image(
    original_image: np.ndarray, 
    edge_image: np.ndarray
) -> np.ndarray:
    """Create a side-by-side comparison of original and edge-detected images.
    
    Args:
        original_image: Original input image
        edge_image: Edge-detected image
        
    Returns:
        Combined comparison image
    """
    edge_image_colored = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    combined_image = np.hstack((original_image, edge_image_colored))
    return combined_image

def create_overlay_image(
    base_image: np.ndarray,
    detected_lines: List
) -> np.ndarray:
    """Create an overlay image showing straight lines on base image.
    
    Args:
        base_image: Base image (can be original or edge-detected)
        detected_lines: List of detected straight lines
        
    Returns:
        Overlay image with lines in red
    """
    if len(base_image.shape) == 2:
        overlay_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_image = base_image.copy()
    
    for line in detected_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(overlay_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return overlay_image

def save_edge_image(edge_image: np.ndarray, output_path: str) -> None:
    """Save the edge-detected image to file.
    
    Args:
        edge_image: Edge-detected image to save
        output_path: Path where to save the image
    """
    try:
        cv2.imwrite(output_path, edge_image)
        print(f"Image saved to: {output_path}")
    except Exception as error:
        print(f"Error saving image: {error}")

def analyze_edge_detection_results(
    original_image: np.ndarray, 
    edge_image: np.ndarray
) -> None:
    """Analyze and print statistics about the edge detection results.
    
    Args:
        original_image: Original input image
        edge_image: Edge-detected image
    """
    total_pixels = edge_image.shape[0] * edge_image.shape[1]
    edge_pixels = np.count_nonzero(edge_image)
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    print("Edge Detection Analysis:")
    print(f"  Original image dimensions: {original_image.shape[:2]}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Edge pixels detected: {edge_pixels:,}")
    print(f"  Edge pixel percentage: {edge_percentage:.2f}%")
    print(f"  Edge density: {edge_pixels / 1000:.1f} edges per 1000 pixels")

def analyze_line_detection_results(
    lines_image: np.ndarray,
    detected_lines: List
) -> None:
    """Analyze and print statistics about the line detection results.
    
    Args:
        lines_image: Straight lines image
        detected_lines: List of detected lines
    """
    total_pixels = lines_image.shape[0] * lines_image.shape[1]
    line_pixels = np.count_nonzero(lines_image)
    line_percentage = (line_pixels / total_pixels) * 100
    
    print("Line Detection Analysis:")
    print(f"  Total straight lines detected: {len(detected_lines)}")
    print(f"  Line pixels: {line_pixels:,}")
    print(f"  Line pixel percentage: {line_percentage:.2f}%")
    print(f"  Average line length: {line_pixels / max(len(detected_lines), 1):.1f} pixels per line")

def main() -> None:
    """Main function to perform color filtering, edge detection, and line detection."""
    image_path = r"/home/eagle/EagleEye-Object-Detection/src/main_operations/modules/edge_detection/Untitled.png"
    
    target_color_1 = (33, 33, 37) # Reef
    target_color_2 = (123, 113, 107) # Wall
    color_tolerance = 20
    
    print("Starting enhanced edge detection process with color filtering...")
    print(f"Processing image: {Path(image_path).name}")
    print(f"Target colors: {target_color_1} and {target_color_2} (BGR format)")
    
    start_time_load = time.time()
    original_image = load_image(image_path)
    if original_image is None:
        return
    end_time_load = time.time()
    load_duration_ms = (end_time_load - start_time_load) * 1000
    print(f"Image loading took: {load_duration_ms:.2f} ms")
    
    print(f"Successfully loaded image with shape: {original_image.shape}")
    
    output_directory = Path(image_path).parent
    
    print("\n=== Step 1: Color Squashing ===")
    start_time_squash = time.time()
    color_squashed_image = squash_colors_to_targets(
        original_image, target_color_1, target_color_2, color_tolerance
    )
    end_time_squash = time.time()
    squash_duration_ms = (end_time_squash - start_time_squash) * 1000
    print(f"Color squashing took: {squash_duration_ms:.2f} ms")
    
    color_squashed_output_path = str(output_directory / "01_color_squashed.png")
    save_edge_image(color_squashed_image, color_squashed_output_path)
    
    print("\n=== Step 2: Edge Detection on Color-Squashed Image ===")
    start_time_edge = time.time()
    edge_detected_image = apply_canny_edge_detection(color_squashed_image)
    end_time_edge = time.time()
    edge_duration_ms = (end_time_edge - start_time_edge) * 1000
    print(f"Edge detection took: {edge_duration_ms:.2f} ms")
    analyze_edge_detection_results(color_squashed_image, edge_detected_image)
    
    edge_output_path = str(output_directory / "02_edge_detection.png")
    save_edge_image(edge_detected_image, edge_output_path)
    
    print("\n=== Step 3: Line Detection ===")
    start_time_line = time.time()
    straight_lines_image, detected_lines = detect_straight_lines(edge_detected_image)
    end_time_line = time.time()
    line_duration_ms = (end_time_line - start_time_line) * 1000
    print(f"Line detection took: {line_duration_ms:.2f} ms")
    analyze_line_detection_results(straight_lines_image, detected_lines)
    
    lines_output_path = str(output_directory / "03_line_detection.png")
    save_edge_image(straight_lines_image, lines_output_path)
    
    print("\n=== Step 4: Creating Overlay Images ===")
    start_time_overlay = time.time()
    original_with_lines_overlay = create_overlay_image(original_image, detected_lines)
    original_overlay_path = str(output_directory / "04_original_with_lines.png")
    save_edge_image(original_with_lines_overlay, original_overlay_path)
    
    edges_with_lines_overlay = create_overlay_image(edge_detected_image, detected_lines)
    edges_overlay_path = str(output_directory / "05_edges_with_lines.png")
    save_edge_image(edges_with_lines_overlay, edges_overlay_path)
    end_time_overlay = time.time()
    overlay_duration_ms = (end_time_overlay - start_time_overlay) * 1000
    print(f"Overlay image creation took: {overlay_duration_ms:.2f} ms")
    
    print("\n=== Process Summary ===")
    print("Enhanced edge detection process completed successfully!")
    print(f"All results saved in: {output_directory}")
    print("\nGenerated files:")
    print("  01_color_squashed.png - Color squashed to target colors frame")
    print("  02_edge_detection.png - Edge detection frame")
    print("  03_line_detection.png - Line detection frame")
    print("  04_original_with_lines.png - Original frame with straight lines overlay")
    print("  05_edges_with_lines.png - Edge detection frame with straight lines overlay")
    
    print(f"\nUsed target colors: {target_color_1} and {target_color_2}")
    print(f"Color tolerance: {color_tolerance}")
    print(f"Total lines detected: {len(detected_lines)}")

if __name__ == "__main__":
    main()
