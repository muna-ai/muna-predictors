#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "rich", "torchvision"]
# ///

from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import array, ndarray
from onnxruntime import InferenceSession
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms.v2 import functional as F
from typing import Annotated

KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Rect(BaseModel):
    x: float = Field(description="Normalized minimum point X coordinate.")
    y: float = Field(description="Normalized minimum point Y coordinate.")
    width: float = Field(description="Normalized width.")
    height: float = Field(description="Normalized height.")

class Keypoint(BaseModel):
    x: float = Field("Normalized X position.")
    y: float = Field("Normalized Y position.")
    score: float = Field("Confidence score in range [0, 1].")

class Pose(BaseModel):
    score: float = Field(description="Pose confidence score in range [0, 1].")
    rect: Rect = Field(description="Detected person normalized bounding (x_min,y_min,x_max,y_max) rectangle.")
    nose: Keypoint = Field(description="Nose normalized coordinates and score.")
    left_eye: Keypoint = Field(description="Left eye normalized coordinates and score.")
    right_eye: Keypoint = Field(description="Right eye normalized coordinates and score.")
    left_ear: Keypoint = Field(description="Left ear normalized coordinates and score.")
    right_ear: Keypoint = Field(description="Right ear normalized coordinates and score.")
    left_shoulder: Keypoint = Field(description="Left shoulder normalized coordinates and score.")
    right_shoulder: Keypoint = Field(description="Right shoulder normalized coordinates and score.")
    left_elbow: Keypoint = Field(description="Left elbow normalized coordinates and score.")
    right_elbow: Keypoint = Field(description="Right elbow normalized coordinates and score.")
    left_wrist: Keypoint = Field(description="Left wrist normalized coordinates and score.")
    right_wrist: Keypoint = Field(description="Right wrist normalized coordinates and score.")
    left_hip: Keypoint = Field(description="Left hip normalized coordinates and score.")
    right_hip: Keypoint = Field(description="Right hip normalized coordinates and score.")
    left_knee: Keypoint = Field(description="Left knee normalized coordinates and score.")
    right_knee: Keypoint = Field(description="Right knee normalized coordinates and score.")
    left_ankle: Keypoint = Field(description="Left ankle normalized coordinates and score.")
    right_ankle: Keypoint = Field(description="Right ankle normalized coordinates and score.")

model_path = hf_hub_download("Xenova/movenet-multipose-lightning", "onnx/model.onnx")
model = InferenceSession(model_path)

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=model, model_path=model_path)
    ]
)
def movenet_multipose(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    min_score: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.3
) -> Annotated[
    list[Pose],
    Parameter.BoundingBoxes(description="Detected poses.")
]:
    """
    Detect poses in an image with MoveNet Multipose.
    """
    # Preprocess image
    image = F.resize(image, [192, 192])
    image = image.convert("RGB")
    image_tensor = array(image).astype("float32")
    # Run model
    logits = model.run(None, { "input": image_tensor[None] })[0] # (1,6,56)
    # Parse poses
    valid_pose_mask = logits[0,:,55] >= min_score
    valid_pose_data = logits[0,valid_pose_mask] # (N,56)
    poses = [_parse_pose(data) for data in valid_pose_data]
    # Return
    return poses

def _parse_pose(data: ndarray) -> Pose:
    """
    Parse a pose vector with shape (56,)
    """
    pose_dict = {
        "score": data[55],
        "rect": {
            "x": data[52],
            "y": data[51],
            "width": data[54],
            "height": data[53]
        }
    }
    for idx, keypoint in enumerate(KEYPOINTS):
        kp_x = data[3 * idx + 1]
        kp_y = data[3 * idx]
        kp_score = data[3 * idx + 2]
        pose_dict[keypoint] = { "x": kp_x, "y": kp_y, "score": kp_score }
    return Pose(**pose_dict)

if __name__ == "__main__":
    from rich import print_json
    # Predict
    image_path = Path(__file__).parent / "demo" / "metro.jpg"
    image = Image.open(image_path)
    poses = movenet_multipose(image)
    # Print poses
    print_json(data=[pose.model_dump() for pose in poses])