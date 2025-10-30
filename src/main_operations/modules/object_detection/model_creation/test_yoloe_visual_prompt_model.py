import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")

# Define visual prompts based on a separate reference image
visual_prompts = dict(
    bboxes=np.array(
        [[432, 682, 461, 739], [401, 596, 490, 675]]
    ),  # Box enclosing person
    cls=np.array([0, 1]),  # ID to be assigned for person
)

# Run prediction on a different image, using reference image to guide what to look for
model.predict(
    r"E:\Ceph-Mirror\Python-Files\Projects\FIRST-Note-Detection\src\main_operations\modules\object_detection\model_creation\visual_prompts\random_frame.jpg",  # Target image for detection
    refer_image=r"E:\Ceph-Mirror\Python-Files\Projects\FIRST-Note-Detection\src\main_operations\modules\object_detection\model_creation\visual_prompts\random_frame.jpg",  # Reference image used to get visual prompts
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    show=False,
    show_conf=False,
    conf=0.5,
)

try:
    model.export(format="onnx")
    print("Model exported successfully to ONNX format")
except Exception as export_error:
    print(f"Warning: Failed to export model to ONNX format: {export_error}")
    print("Continuing without export...")
