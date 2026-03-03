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
from torchvision.ops import box_convert
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from typing import Annotated
from ultralytics import YOLO

class Detection(BaseModel):
    x_center: float = Field(description="Normalized bounding box center X-coordinate.")
    y_center: float = Field(description="Normalized bounding box center Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

# Instantiate model
yolo = YOLO("yolo26n.pt")
model: Module = yolo.model.eval()
labels: dict[int, str] = model.names

# Dry run the model to prepare for export
INPUT_SIZE = 640
model_args = [randn(1, 3, INPUT_SIZE, INPUT_SIZE)]
model(*model_args)

# Define predictor
@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ultralytics")
        .pip_install("opencv-python-headless"),
    metadata=[
        OnnxRuntimeInferenceMetadata(model=model, model_args=model_args)
    ]
)
@inference_mode()
def yolo26_nano(
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
) -> Annotated[
    list[Detection],
    Parameter.BoundingBoxes(description="Detected objects.")
]:
    """
    Detect objects in an image with YOLO26 (nano).
    """
    # Preprocess
    image_tensor, scale_factors = _preprocess_image(image, input_size=INPUT_SIZE)
    # Run model
    model_outputs = model(image_tensor[None])
    # YOLO26 outputs (1, 300, 6) with [x1, y1, x2, y2, confidence, class_id]
    predictions: Tensor = model_outputs[0][0]   # (300, 6)
    boxes_xyxy = predictions[:, :4]             # (300, 4)
    scores = predictions[:, 4]                  # (300,)
    class_ids = predictions[:, 5].int()         # (300,)
    # Filter by confidence
    confidence_mask = scores >= min_confidence
    filtered_boxes = boxes_xyxy[confidence_mask] * scale_factors
    filtered_scores = scores[confidence_mask]
    filtered_class_ids = class_ids[confidence_mask]
    # Check if any detections remain
    if len(filtered_boxes) == 0:
        return []
    # Convert xyxy to cxcywh
    filtered_boxes_cxcywh = box_convert(
        filtered_boxes,
        in_fmt="xyxy",
        out_fmt="cxcywh"
    )
    # Create detection objects
    detections = [
        _create_detection(box, class_id=class_id, score=score)
        for box, class_id, score
        in zip(filtered_boxes_cxcywh, filtered_class_ids, filtered_scores)
    ]
    # Return
    return detections

def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int,
    ensure_multiple_of: int=32,
    fill: int=114
) -> tuple[Tensor, Tensor]:
    """
    Preprocess an image for inference by downscaling and padding it to have a square aspect.
    """
    # Compute scaled size
    image_width, image_height = image.size
    ratio = min(input_size / image_width, input_size / image_height)
    scaled_width = int(image_width * ratio)
    scaled_height = int(image_height * ratio)
    # Compute padding
    x_pad = (input_size - scaled_width) % ensure_multiple_of
    y_pad = (input_size - scaled_height) % ensure_multiple_of
    padding = [0, 0, x_pad, y_pad]
    # Downscale and pad image
    image = image.convert("RGB")
    image = F.resize(image, [scaled_height, scaled_width])
    image = F.pad(image, padding, fill=fill)
    # Create tensors
    image_tensor = F.to_tensor(image)
    scaled_sizes = tensor([scaled_width, scaled_height, scaled_width, scaled_height])
    # Return
    return image_tensor, scaled_sizes.reciprocal()

def _create_detection(
    box: Tensor,
    *,
    class_id: Tensor,
    score: Tensor
) -> Detection:
    """
    Create a detection object given raw detection tensors.
    """
    label = labels[class_id.item()]
    detection = Detection(
        x_center=box[0].item(),
        y_center=box[1].item(),
        width=box[2].item(),
        height=box[3].item(),
        label=label,
        confidence=score.item()
    )
    return detection

def _visualize_detections(
    image: Image.Image,
    detections: list[Detection]
) -> Image.Image:
    """
    Visualize detection results on an image.
    """
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    boxes_cxcywh = tensor([[
        detection.x_center * image.width,
        detection.y_center * image.height,
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
    return F.to_pil_image(result_tensor)

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect objects
    image_path = Path(__file__).parent / "demo" / "vehicles.jpg"
    image = Image.open(image_path)
    detections = yolo26_nano(image)
    # Print detections
    print_json(data=[det.model_dump() for det in detections])
    # Show annotated image
    annotated_image = _visualize_detections(image, detections)
    annotated_image.show()
