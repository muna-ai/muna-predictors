#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "opencv-python-headless", "tensorflow", "torchvision"]
# ///

from cv2 import getRectSubPix, getRotationMatrix2D, warpAffine, BORDER_CONSTANT, INTER_LINEAR
from muna import Parameter, Sandbox, compile
from muna.beta import TFLiteInterpreterMetadata
from numpy import array
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from shutil import unpack_archive
from tensorflow import lite
from torch import cos, from_numpy, sin, stack, tensor, Tensor
from torch.hub import download_url_to_file
from typing import Annotated

class Rect(BaseModel):
    x_center: float = Field(description="Normalized bounding box X-coordinate.")
    y_center: float = Field(description="Normalized bounding box Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")

class RotatedRect(Rect):
    rotation: float = Field(description="Bounding box rect.")

class Keypoint(BaseModel):
    position: tuple[float, float, float] = Field(description="Normalized keypoint position.")
    world_position: tuple[float, float, float] = Field(description="Keypoint position in world space.")
    visibility: float = Field(description="Keypoint visibility.")
    presence: float = Field(description="Keypoint presence.")

class FullBodyPose(BaseModel):
    rect: Rect = Field(description="Pose rectangle.")
    keypoints: dict[str, Keypoint] = Field(description="Pose keypoints.")

# Download landmark model (there are 3 variants)
# Options: pose_landmarker_lite, pose_landmarker_full, pose_landmarker_heavy
MODEL_VARIANT = "pose_landmarker_lite"  # Change this to switch models
model_path = Path("pose_landmarks_detector.tflite")
if not model_path.exists():
    model_task_url = (
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"{MODEL_VARIANT}/float16/latest/{MODEL_VARIANT}.task"
    )
    model_task_path = Path(Path(model_task_url).name)
    download_url_to_file(model_task_url, model_task_path)
    unpack_archive(model_task_path, format="zip")

# Load the model
interpreter = lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Get I/O tensor indices
_get_tensor_idx = lambda details, name: next(detail for detail in details if detail["name"] == name)["index"]
interpreter_image_idx = _get_tensor_idx(interpreter.get_input_details(), "input_1")
interpreter_landmarks_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity")
interpreter_presence_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_1")
interpreter_landmarks_3d_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_4")

KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("opencv-python-headless", "tensorflow"),
    metadata=[
        TFLiteInterpreterMetadata(interpreter=interpreter, model_path=model_path),
    ]
)
def blazepose_landmarks_lite(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")],
    roi: Annotated[RotatedRect, Parameter.Generic(description="Pose RoI.")]
) -> Annotated[FullBodyPose, Parameter.Generic(description="Full body pose.")]:
    """
    Detect a body pose an from RoI with BlazePose Landmarks (lite).
    """
    # Create ROI image
    image = image.convert("RGB")
    roi_image = _create_roi_image(image, roi)
    roi_tensor = array(roi_image).astype("float32") / 255
    # Run model
    interpreter.set_tensor(interpreter_image_idx, roi_tensor[None])
    interpreter.invoke()
    raw_landmarks = from_numpy(interpreter.get_tensor(interpreter_landmarks_idx))
    raw_presence = from_numpy(interpreter.get_tensor(interpreter_presence_idx))
    raw_landmarks_3d = from_numpy(interpreter.get_tensor(interpreter_landmarks_3d_idx))
    # Process outputs - flatten to 1D first, then reshape
    raw_landmarks = raw_landmarks[0].view(-1, 5) # (x, y, z, visibility, presence)
    raw_landmarks_3d = raw_landmarks_3d[0].view(-1, 3) # (x, y, z)
    # Take first 33 landmarks (standard MediaPipe pose landmarks)
    num_keypoints = len(KEYPOINT_NAMES)
    raw_landmarks = raw_landmarks[:num_keypoints]
    raw_landmarks_3d = raw_landmarks_3d[:num_keypoints]
    # Extract normalized 2D coordinates (x, y) from landmarks
    landmarks_2d = raw_landmarks[:,:2] / 256.0
    # Extract visibility and presence probabilities from the landmarks tensor
    visibility = raw_landmarks[:,3].sigmoid()
    presence = raw_landmarks[:,4].sigmoid()
    # Create keypoints
    transformed_landmarks = _transform_landmarks_from_roi(landmarks_2d, roi)
    keypoints = {
        name: Keypoint(
            position=(
                transformed_landmarks[i, 0].item(),
                transformed_landmarks[i, 1].item(),
                raw_landmarks[i, 2].item() / 256.0
            ),
            world_position=raw_landmarks_3d[i].tolist(),
            visibility=visibility[i].item(),
            presence=presence[i].item()
        )
        for i, name in enumerate(KEYPOINT_NAMES)
    }
    # Create pose rectangle (use detection's body rectangle)
    pose_rect = Rect(
        x_center=roi.x_center,
        y_center=roi.y_center,
        width=roi.width,
        height=roi.height
    )
    # Return
    return FullBodyPose(rect=pose_rect, keypoints=keypoints)

def _transform_landmarks_from_roi(landmarks: Tensor, roi: RotatedRect) -> Tensor:
    """
    Transform landmarks from ROI space back to original image coordinates.
    Uses MediaPipe's landmark projection formula.
    """
    angle_rad = tensor(roi.rotation) * 3.14159265359 / 180.0
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    # Center landmarks (landmarks are in [0, 1] space)
    x_centered = landmarks[:,0] - 0.5
    y_centered = landmarks[:,1] - 0.5
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
    *,
    size: int=256
) -> Image.Image:
    """
    Extracts an upright region of interest around the detected body rectangle.
    """
    # Compute rotation matrix
    center_x = detection.x_center * image.width
    center_y = detection.y_center * image.height
    box_size_px = max(
        int(round(detection.width * image.width)),
        int(round(detection.height * image.height))
    )
    rotation_matrix = getRotationMatrix2D(
        center=(center_x, center_y),
        angle=detection.rotation,
        scale=size / box_size_px,
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
    roi = getRectSubPix(rotated, (size, size), (center_x, center_y))
    # Return
    return Image.fromarray(roi)

def _visualize_full_body_poses(
    image: Image.Image,
    poses: list[FullBodyPose],
    *,
    min_visibility: float=0.5
) -> Image.Image:
    """
    Visualize full body pose landmarks on an image.
    """
    from cv2 import circle, cvtColor, line, COLOR_BGR2RGB, COLOR_RGB2BGR

    # MediaPipe pose connections (landmark indices to connect)
    POSE_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        # Torso
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        # Legs
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32)
    ]
    image_rgb = image.convert("RGB")
    canvas = cvtColor(array(image_rgb), COLOR_RGB2BGR)
    height, width, _ = canvas.shape
    for pose in poses:
        # Convert keypoints to pixel coordinates
        keypoint_coords = {}
        for name, keypoint in pose.keypoints.items():
            if keypoint.visibility >= min_visibility:
                x = int(keypoint.position[0] * width)
                y = int(keypoint.position[1] * height)
                keypoint_coords[name] = (x, y)
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_name = KEYPOINT_NAMES[start_idx]
            end_name = KEYPOINT_NAMES[end_idx]
            if start_name in keypoint_coords and end_name in keypoint_coords:
                start_pos = keypoint_coords[start_name]
                end_pos = keypoint_coords[end_name]
                line(canvas, start_pos, end_pos, color=(0, 255, 0), thickness=2)
        # Draw keypoints
        for name, (x, y) in keypoint_coords.items():
            circle(canvas, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    result_rgb = cvtColor(canvas, COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print_json
    # Infer full body pose
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    roi = RotatedRect(
        x_center=0.6124255061149597,
        y_center=0.520103394985199,
        width=0.8662170171737671,
        height=1.293550729751587,
        rotation=9.145028114318848
    )
    pose = blazepose_landmarks_lite(image, roi)
    # Visualize
    print_json(data=TypeAdapter(FullBodyPose).dump_python(pose))
    annotated_image = _visualize_full_body_poses(image, [pose])
    annotated_image.show()