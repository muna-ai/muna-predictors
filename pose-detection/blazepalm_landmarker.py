#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["ai-edge-litert", "muna", "opencv-python-headless", "torchvision"]
# ///

from ai_edge_litert.interpreter import Interpreter
from cv2 import getRectSubPix, getRotationMatrix2D, warpAffine, BORDER_CONSTANT, INTER_LINEAR
from muna import compile, Parameter, Sandbox
from muna.beta import TFLiteInterpreterMetadata
from numpy import array
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from shutil import unpack_archive
from torch import cos, from_numpy, sin, stack, tensor, Tensor
from torch.hub import download_url_to_file
from typing import Annotated, Literal

class Rect(BaseModel):
    x_center: float = Field(description="Normalized bounding box X-coordinate.")
    y_center: float = Field(description="Normalized bounding box Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")

class RotatedRect(Rect):
    rotation: float = Field(description="Bounding box rotation in degrees.")

class HandLandmark(BaseModel):
    position: tuple[float, float, float] = Field(description="Normalized landmark position (x, y, z).")
    world_position: tuple[float, float, float] = Field(description="Landmark position in world space (x, y, z).")

class HandPose(BaseModel):
    rect: Rect = Field(description="Hand bounding rectangle.")
    landmarks: list[HandLandmark] = Field(description="Hand landmarks (21 landmarks in MediaPipe order).")
    handedness: Literal["left", "right"] = Field(description="Detected handedness.")
    score: float = Field(description="Hand presence score.")

# Download hand landmarker model
model_task_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
landmarks_model_path = Path("hand_landmarks_detector.tflite")
if not landmarks_model_path.exists():
    model_task_path = Path("hand_landmarker.task")
    print("Downloading hand_landmarker.task...")
    download_url_to_file(model_task_url, model_task_path)
    unpack_archive(model_task_path, format="zip")

# Load the landmark model
interpreter = Interpreter(str(landmarks_model_path))
interpreter.allocate_tensors()

# Get I/O tensor indices
_get_tensor_idx = lambda details, name: next(detail for detail in details if detail["name"] == name)["index"]
interpreter_image_idx = _get_tensor_idx(interpreter.get_input_details(), "input_1")
interpreter_landmarks_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity")
interpreter_score_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_1")
interpreter_handedness_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_2")
interpreter_world_landmarks_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_3")

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ai-edge-litert", "opencv-python-headless"),
    metadata=[
        TFLiteInterpreterMetadata(
            interpreter=interpreter,
            model_path=landmarks_model_path
        ),
    ]
)
def blazepalm_landmarker(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    roi: Annotated[
        RotatedRect,
        Parameter.Generic(description="Hand region of interest from palm detector.")
    ]
) -> Annotated[
    HandPose,
    Parameter.Generic(description="Detected hand pose with 21 landmarks.")
]:
    """
    Detect hand landmarks from a region of interest using the MediaPipe hand landmarker.
    """
    # Create ROI image
    image = image.convert("RGB")
    roi_image = _create_roi_image(image, roi)
    roi_tensor = array(roi_image).astype("float32") / 255.0
    # Run model
    interpreter.set_tensor(interpreter_image_idx, roi_tensor[None])
    interpreter.invoke()
    raw_landmarks = from_numpy(interpreter.get_tensor(interpreter_landmarks_idx))
    raw_score = from_numpy(interpreter.get_tensor(interpreter_score_idx))
    raw_handedness = from_numpy(interpreter.get_tensor(interpreter_handedness_idx))
    raw_world_landmarks = from_numpy(interpreter.get_tensor(interpreter_world_landmarks_idx))
    # Process landmarks: [1, 63] -> [21, 3]
    landmarks_2d = raw_landmarks[0].view(21, 3)
    world_landmarks = raw_world_landmarks[0].view(21, 3)
    # Normalize to [0, 1] range
    positions_xy = landmarks_2d[:, :2] / 224.0
    positions_z = landmarks_2d[:, 2:3] / 224.0
    # Determine handedness (sigmoid of raw logit) and score
    handedness_score = float(raw_handedness[0, 0].sigmoid().item())
    handedness = "right" if handedness_score > 0.5 else "left"
    score = float(raw_score[0, 0].sigmoid().item())
    # Transform landmarks back to original image coordinates
    transformed_landmarks = _transform_landmarks_from_roi(positions_xy, roi)
    # Build landmarks list
    landmarks = [HandLandmark(
        position=(
            transformed_landmarks[i, 0].item(),
            transformed_landmarks[i, 1].item(),
            positions_z[i, 0].item()
        ),
        world_position=world_landmarks[i].tolist()
    ) for i in range(21)]
    # Create hand rect from ROI
    hand_rect = Rect(
        x_center=roi.x_center,
        y_center=roi.y_center,
        width=roi.width,
        height=roi.height
    )
    # Return
    return HandPose(
        rect=hand_rect,
        landmarks=landmarks,
        handedness=handedness,
        score=score
    )

def _transform_landmarks_from_roi(landmarks: Tensor, roi: RotatedRect) -> Tensor:
    """
    Transform landmarks from ROI space back to original image coordinates.
    """
    angle_rad = tensor(roi.rotation) * 3.14159265359 / 180.0
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    # Center landmarks (landmarks are in [0, 1] space)
    x_centered = landmarks[:, 0] - 0.5
    y_centered = landmarks[:, 1] - 0.5
    # Apply rotation
    x_rotated = cos_angle * x_centered - sin_angle * y_centered
    y_rotated = sin_angle * x_centered + cos_angle * y_centered
    # Scale by rect dimensions and translate to center
    x_final = x_rotated * roi.width + roi.x_center
    y_final = y_rotated * roi.height + roi.y_center
    return stack([x_final, y_final], dim=1)

def _create_roi_image(
    image: Image.Image,
    detection: RotatedRect,
) -> Image.Image:
    """
    Extracts an upright region of interest around the detected hand.
    """
    # Compute rotation matrix
    roi_size = 224
    center_x = detection.x_center * image.width
    center_y = detection.y_center * image.height
    box_size_px = max(
        int(round(detection.width * image.width)),
        int(round(detection.height * image.height))
    )
    rotation_matrix = getRotationMatrix2D(
        center=(center_x, center_y),
        angle=detection.rotation,
        scale=roi_size / box_size_px,
    )
    # Sample
    rotated = warpAffine(
        array(image),
        rotation_matrix,
        image.size,
        flags=INTER_LINEAR,
        borderMode=BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    roi = getRectSubPix(rotated, (roi_size, roi_size), (center_x, center_y))
    # Return
    return Image.fromarray(roi)

def _visualize_hand_landmarks(
    image: Image.Image,
    poses: list[HandPose],
) -> Image.Image:
    """
    Visualize hand landmarks on an image.
    """
    from cv2 import circle, cvtColor, line, putText, COLOR_BGR2RGB, COLOR_RGB2BGR, FONT_HERSHEY_SIMPLEX, LINE_AA
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    image_rgb = image.convert("RGB")
    canvas = cvtColor(array(image_rgb), COLOR_RGB2BGR)
    height, width, _ = canvas.shape
    for pose in poses:
        # Convert landmarks to pixel coordinates
        landmark_coords = {}
        for i, lm in enumerate(pose.landmarks):
            x = int(lm.position[0] * width)
            y = int(lm.position[1] * height)
            landmark_coords[i] = (x, y)
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx in landmark_coords and end_idx in landmark_coords:
                start_pos = landmark_coords[start_idx]
                end_pos = landmark_coords[end_idx]
                line(canvas, start_pos, end_pos, color=(0, 255, 0), thickness=2)
        # Draw landmarks
        for idx, (x, y) in landmark_coords.items():
            circle(canvas, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
        # Add handedness label
        wrist = landmark_coords.get(0)
        if wrist:
            label = f"{pose.handedness} ({pose.score:.2f})"
            putText(canvas, label, (wrist[0], wrist[1] - 10), FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, LINE_AA)
    result_rgb = cvtColor(canvas, COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print_json
    # Test with a known ROI (from hand detector)
    image_path = Path(__file__).parent / "demo" / "palms.jpg"
    image = Image.open(image_path)
    # Use a default ROI covering center of image for testing
    roi = RotatedRect(
        x_center=0.5,
        y_center=0.5,
        width=0.8,
        height=0.8,
        rotation=0.0
    )
    pose = blazepalm_landmarker(image, roi)
    # Visualize
    print_json(data=TypeAdapter(HandPose).dump_python(pose))
    annotated_image = _visualize_hand_landmarks(image, [pose])
    annotated_image.save("hand_landmarks_result.jpg")
    print(f"Handedness: {pose.handedness}, Score: {pose.score:.3f}")
