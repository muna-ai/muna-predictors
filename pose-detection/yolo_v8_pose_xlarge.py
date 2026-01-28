#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "rich", "torchvision", "ultralytics"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from pydantic import BaseModel, Field
from torch import inference_mode, randn, tensor, Tensor
from torch.nn import Module
from torchvision.ops import box_convert, nms
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from typing import Annotated
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye",  "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Keypoint(BaseModel):
    x: float = Field(description="Normalized x-coordinate of the keypoint.")
    y: float = Field(description="Normalized y-coordinate of the keypoint.")
    label: str = Field(description="Keypoint label.")
    confidence: float = Field(description="Normalized keypoint confidence score.")

class Pose(BaseModel):
    center_x: float = Field(description="Normalized bounding box center x coordinate.")
    center_y: float = Field(description="Normalized bounding box center y coordinate.")
    width: float = Field(description="Normalized bounding box width")
    height: float = Field(description="Normalized bounding box width")
    label: str = Field(description="Detection label. Always 'person'.")
    confidence: float = Field(description="Detection confidence score.")
    keypoints: list[Keypoint] = Field(description="Pose keypoints.")

# Create the YOLO Model
yolo = YOLO("yolov8x-pose.pt")
model: Module = yolo.model.eval()
labels: dict[int, str] = model.names

# Dry run the model for export
INPUT_SIZE = 640
model_args = [randn(1, 3, INPUT_SIZE, INPUT_SIZE)]
model(*model_args)

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ultralytics")
        .pip_install("opencv-python-headless"),
    metadata=[
        OnnxRuntimeInferenceMetadata(model=model, model_args=model_args)
    ]
)
@inference_mode()
def yolo_v8_pose_xlarge(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.25,
    max_iou: Annotated[float, Parameter.Numeric(
        description="Maximum intersection-over-union score before discarding smaller detections.",
        min=0.,
        max=1.
    )]=0.25
) -> Annotated[
    list[Pose],
    Parameter.BoundingBoxes(description="Detected poses.")
]:
    """
    Detect poses in an image with YOLO-v8 (xlarge).
    """
    image_tensor, box_scale_factors, keypoint_scale_factors = _preprocess_image(image, input_size=640)
    model_outputs: Tensor = model(image_tensor[None])[0]    # (1,4+1+P,8400)
    logits = model_outputs[0].T                             # (8400,4+1+P)
    boxes_cxcywh = logits[:,:4]                             # (8400,4)
    max_scores = logits[:,4]                                # (8400,)
    keypoints = logits[:,5:].reshape(-1, 17, 3)             # (8400,51) -> (8400,17,3)
    # Filter by score
    confidence_mask = max_scores >= min_confidence
    filtered_boxes = boxes_cxcywh[confidence_mask] * box_scale_factors
    filtered_scores = max_scores[confidence_mask]
    filtered_keypoints = keypoints[confidence_mask] * keypoint_scale_factors
    # Check if any detections remain
    if len(filtered_boxes) == 0:
        return []
    # Apply NMS
    boxes_xyxy = box_convert(
        filtered_boxes,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    )
    keep_indices = nms(
        boxes_xyxy,
        scores=filtered_scores,
        iou_threshold=max_iou
    )
    final_boxes = filtered_boxes[keep_indices]
    final_scores = filtered_scores[keep_indices]
    final_keypoints = filtered_keypoints[keep_indices]
    # Create pose objects
    poses = [
        _create_pose(box, kps, confidence)
        for box, kps, confidence
        in zip(final_boxes, final_keypoints, final_scores)
    ]
    # Return
    return poses

def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Preprocess an image for inference by downscaling and padding it to have a square aspect.
    """
    # Compute scaled size and padding
    image_width, image_height = image.size
    ratio = min(input_size / image_width, input_size / image_height)
    scaled_width = int(image_width * ratio)
    scaled_height = int(image_height * ratio)
    image_padding = [0, 0, input_size - scaled_width, input_size - scaled_height]
    # Downscale and pad image
    image = image.convert("RGB")
    image = F.resize(image, [scaled_height, scaled_width])
    image = F.pad(image, image_padding, fill=114)
    # Create tensors
    image_tensor = F.to_tensor(image)
    box_scale_factors = tensor([scaled_width, scaled_height, scaled_width, scaled_height]).reciprocal()
    keypoint_scale_factors = tensor([scaled_width, scaled_height, 1.]).reciprocal()
    # Return
    return image_tensor, box_scale_factors, keypoint_scale_factors

def _create_pose(
    box: Tensor,
    keypoints: Tensor,
    confidence: Tensor
) -> Pose:
    """
    Create a `Pose` object from raw pose tensor data.
    """
    keypoint_data = [Keypoint(
        x=row[0].item(),
        y=row[1].item(),
        confidence=row[2].item(),
        label=KEYPOINT_NAMES[idx]
    ) for idx, row in enumerate(keypoints)]
    pose = Pose(
        center_x=box[0].item(),
        center_y=box[1].item(),
        width=box[2].item(),
        height=box[3].item(),
        label=labels[0],
        confidence=confidence.item(),
        keypoints=keypoint_data
    )
    return pose

def _visualize_poses(
    image: Image.Image,
    detections: list[Pose]
) -> Image.Image:
    """
    Render poses on an image.
    """
    from PIL import ImageDraw
    KEYPOINT_SKELETON = [
        # nose to eyes
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        # eyes to ears
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        # shoulders   
        ("left_shoulder", "right_shoulder"),
        # left arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        # right arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        # shoulders to hips  
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        # hips
        ("left_hip", "right_hip"),
        # left leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        # right leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle")  
    ]
    KEYPOINT_COLOR_MAP = {
        "nose":"red",
        "left_eye": "blue",
        "right_eye": "blue",
        "left_ear": "purple",
        "right_ear": "purple",
        "left_shoulder": "orange",
        "right_shoulder": "orange",
        "left_elbow": "yellow",
        "right_elbow": "yellow",
        "left_wrist": "cyan",
        "right_wrist": "cyan",
        "left_hip": "magenta",
        "right_hip": "magenta",
        "left_knee":"pink",
        "right_knee":"pink",
        "left_ankle": "brown",
        "right_ankle": "brown"
    }
    # Draw bounding boxes
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    boxes_cxcywh = tensor([[
        detection.center_x * image.width,
        detection.center_y * image.height,
        detection.width * image.width,
        detection.height * image.height
    ] for detection in detections])
    boxes_xyxy = box_convert(
        boxes_cxcywh,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    )
    labels = [detection.label for detection in detections]
    result_tensor = draw_bounding_boxes(
        image_tensor,
        boxes=boxes_xyxy,
        labels=labels,
        width=8,
        font="Arial",
        font_size=int(0.015 * image.width)
    )
    # Convert back to PIL for keypoint drawing
    result_image = F.to_pil_image(result_tensor)
    draw = ImageDraw.Draw(result_image)
    # Draw keypoints and skeleton for each detection
    for detection in detections:
        keypoints = detection.keypoints
        # Draw skeleton connections (joints)
        for start_label, end_label in KEYPOINT_SKELETON:
            start_kp = next(kp for kp in keypoints if kp.label == start_label)
            end_kp = next(kp for kp in keypoints if kp.label == end_label)
            start_x = start_kp.x * image.width
            start_y = start_kp.y * image.height
            end_x = end_kp.x * image.width
            end_y = end_kp.y * image.height
            draw.line([(start_x, start_y), (end_x, end_y)], fill="lime", width=3)
        # Draw keypoints
        for keypoint in keypoints:
            x = keypoint.x * image.width
            y = keypoint.y * image.height
            radius = 6
            draw.ellipse(
                [x-radius, y-radius, x+radius, y+radius], 
                fill=KEYPOINT_COLOR_MAP[keypoint.label],
                outline="black",
                width=2
            )
    # Return
    return result_image

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect poses
    image_path = Path(__file__).parent / "demo" / "metro.jpg"
    image = Image.open(image_path)
    poses = yolo_v8_pose_xlarge(image)
    # Print detections
    print(f"Detected {len(poses)} poses:")
    print_json(data=[pose.model_dump() for pose in poses])
    # Show annotated image
    annotated_image = _visualize_poses(image, poses)
    annotated_image.show()