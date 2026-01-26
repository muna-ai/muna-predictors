#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru",
#     "muna",
#     "opencv-python-headless",
#     "packaging",
#     "psutil",
#     "rich",
#     "tabulate",
#     "torchvision"
# ]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from pydantic import BaseModel, Field
from torch import inference_mode, randn, tensor, Tensor
from torch.hub import load
from torch.nn import Module
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms, box_convert
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from typing import Annotated

class Detection(BaseModel):
    x_center: float = Field(description="Normalized bounding box center X-coordinate.")
    y_center: float = Field(description="Normalized bounding box center Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

# Instantiate model
model: Module = load("Megvii-BaseDetection/YOLOX", "yolox_nano", trust_repo=True)
model.eval()
INPUT_SIZE = 320

# Get COCO detection labels
labels = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
labels = [label for label in labels if label not in ("__background__", "N/A")]

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("loguru", "opencv-python-headless", "psutil", "tabulate"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, INPUT_SIZE, INPUT_SIZE)]
        )
    ]
)
@inference_mode()
def yolox_nano(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.4,
    max_iou: Annotated[float, Parameter.Numeric(
        description="Maximum intersection-over-union score to discard smaller detections.",
        min=0.,
        max=1.
    )]=0.1
) -> Annotated[
    list[Detection],
    Parameter.BoundingBoxes(description="Detected objects.")
]:
    """
    Detect objects in an image with YOLOX (nano).
    """
    # Preprocess image
    image_tensor, scale_factors = _preprocess_image(image, input_size=INPUT_SIZE)
    # Run model
    logits: Tensor = model(image_tensor[None])
    # Parse boxes
    boxes_cxcywh = logits[0,:,:4]
    box_scores = logits[0,:,4,]
    class_logits = logits[0,:,5:]
    class_scores, class_ids = class_logits.max(dim=1)
    box_confidences = box_scores * class_scores
    # Filter
    confidence_mask = box_confidences >= min_confidence
    filtered_boxes = boxes_cxcywh[confidence_mask] * scale_factors
    filtered_confidences = box_confidences[confidence_mask]
    filtered_class_ids = class_ids[confidence_mask]
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
        scores=filtered_confidences,
        iou_threshold=max_iou
    )
    # Select final detections
    final_boxes = filtered_boxes[keep_indices]  # Keep in cxcywh format
    final_confidences = filtered_confidences[keep_indices]
    final_class_ids = filtered_class_ids[keep_indices]
    # Create detection objects
    detections = [
        _create_detection(box, class_id=class_id, score=score)
        for box, class_id, score
        in zip(final_boxes, final_class_ids, final_confidences)
    ]
    # Return
    return detections

def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int
) -> tuple[Tensor, Tensor]:
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
    image_tensor = F.pil_to_tensor(image).float()
    scaled_sizes = tensor([scaled_width, scaled_height, scaled_width, scaled_height])
    # Return
    return image_tensor, scaled_sizes.reciprocal()

def _create_detection(
    box: Tensor,
    *,
    class_id: Tensor,
    score: Tensor
) -> Detection:
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
        font_size=int(0.04 * image.width)
    )
    return F.to_pil_image(result_tensor)

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect objects
    image_path = Path(__file__).parent / "demo" / "vehicles.jpg"
    image = Image.open(image_path)
    detections = yolox_nano(image, min_confidence=0.2)
    # Print detections
    print_json(data=[det.model_dump() for det in detections])
    # Show annotated image
    annotated_image = _visualize_detections(image, detections)
    annotated_image.show()