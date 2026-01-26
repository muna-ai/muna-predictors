#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["faster_coco_eval", "muna", "rich", "scipy", "tensorboard", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from pydantic import BaseModel, Field
from torch import hub, inference_mode, tensor, randn
from torch.nn import Module
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from typing import Annotated

# Load the model
model_name = "rtdetr_r18vd"
model: Module = hub.load("lyuwenyu/RT-DETR", model_name, pretrained=True).eval()
INPUT_SIZE = 640

# Get COCO class names from torchvision
coco_classes = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
coco_classes = [label for label in coco_classes if label not in ("__background__", "N/A")]

class Detection(BaseModel):
    x_min: float = Field(description="Normalized minimum X coordinate.")
    y_min: float = Field(description="Normalized minimum Y coordinate.")
    x_max: float = Field(description="Normalized maximum X coordinate.")
    y_max: float = Field(description="Normalized maximum Y coordinate.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

@compile(
    description="Detect objects in an image with RT-DETR.",
    access="public",
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("faster_coco_eval", "tensorboard", "scipy"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[
                randn(1, 3, INPUT_SIZE, INPUT_SIZE),
                tensor([INPUT_SIZE, INPUT_SIZE]).float()[None]
            ]
        )
    ]
)
@inference_mode()
def rt_detr(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.5
) -> Annotated[
    list[Detection],
    Parameter.BoundingBoxes(description="Detected objects.")
]:
    """
    Detect objects in an image with RT-DETR.
    """
    width, height = image.size
    # Preprocess image
    image = F.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image.convert("RGB")
    # Convert to tensor and add 1 (batch) dimension
    input_tensor = F.to_tensor(image)
    orig_target_sizes = tensor([width, height]).float()
    # Run model
    predictions = model(input_tensor[None], orig_target_sizes[None])
    # Extract results and filter by confidence threshold
    labels, boxes, scores = predictions
    valid_detections = scores[0] > min_confidence
    labels = labels[0][valid_detections]
    boxes = boxes[0][valid_detections]
    scores = scores[0][valid_detections]
    # Check if any detections remain
    if len(labels) == 0:
        return []
    # Normalize coordinates of the detected boxes
    scale_factors = orig_target_sizes.repeat(2).reciprocal()
    boxes_normalized = boxes * scale_factors
    # Create detection objects
    detections = [Detection(
        x_min=boxes_normalized[i,0].item(),
        y_min=boxes_normalized[i,1].item(),
        x_max=boxes_normalized[i,2].item(),
        y_max=boxes_normalized[i,3].item(),
        label=coco_classes[labels[i].item()],
        confidence=scores[i].item()
    ) for i in range(len(labels))]
    # Return
    return detections

def _visualize_detections(
    image: Image.Image,
    detections: list[Detection]
) -> Image.Image:
    """
    Visualize detection results on an image.
    """
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    # Convert detections to tensor format and denormalize
    boxes_xyxy = tensor([[det.x_min, det.y_min, det.x_max, det.y_max] for det in detections])
    boxes_xyxy *= tensor(image.size).repeat(2)    
    # Draw bounding boxes with torchvision
    labels = [f"{det.label}: {det.confidence:.2f}" for det in detections]
    result_tensor = draw_bounding_boxes(
        image_tensor,
        boxes=boxes_xyxy,
        labels=labels,
        width=3,
        font="Arial",
        font_size=int(0.03 * image.width)
    )
    # Convert back to PIL Image
    return F.to_pil_image(result_tensor)

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect objects
    image_path = Path(__file__).parent / "demo" / "vehicles.jpg"
    image = Image.open(image_path)
    detections = rt_detr(image, min_confidence=0.5)
    # Print detections
    print_json(data=[det.model_dump() for det in detections])
    # Render and show the result
    result_image = _visualize_detections(image, detections)
    result_image.show()