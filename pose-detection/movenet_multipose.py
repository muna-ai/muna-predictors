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

class Keypoint(BaseModel):
    x: float = Field("Normalized X position.")
    y: float = Field("Normalized Y position.")
    label: str = Field(description="Keypoint label.")
    confidence: float = Field("Normalized keypoint confidence score.")

class Pose(BaseModel):
    x: float = Field(description="Normalized minimum point X coordinate.")
    y: float = Field(description="Normalized minimum point Y coordinate.")
    width: float = Field(description="Normalized width.")
    height: float = Field(description="Normalized height.")
    score: float = Field(description="Pose confidence score in range [0, 1].")
    keypoints: list[Keypoint] = Field(description="Pose keypoints.")

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
    image_tensor = array(image).astype("int32")
    # Run model
    logits = model.run(None, { "input": image_tensor[None] })[0] # (1,6,56)
    # Parse poses
    valid_pose_mask = logits[0,:,55] >= min_score
    valid_pose_data = logits[0,valid_pose_mask] # (N,56)
    poses = [_create_pose(row) for row in valid_pose_data]
    # Return
    return poses

def _create_pose(data: ndarray) -> Pose:
    """
    Create a `Pose` object from raw pose tensor data.
    """
    keypoints=[Keypoint(
        x=data[3 * idx + 1].item(),
        y=data[3 * idx].item(),
        label=keypoint,
        confidence=data[3 * idx + 2].item()
    ) for idx, keypoint in enumerate(KEYPOINTS)]
    pose = Pose(
        x=data[52].item(),
        y=data[51].item(),
        width=data[54].item(),
        height=data[53].item(),
        score=data[55].item(),
        keypoints=keypoints
    )
    return pose

if __name__ == "__main__":
    from rich import print_json
    # Predict
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    poses = movenet_multipose(image)
    # Print poses
    print_json(data=[pose.model_dump() for pose in poses])