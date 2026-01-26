#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "rfdetr", "rich", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from pydantic import BaseModel, Field
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from torch import inference_mode, randn, tensor, Tensor
from torchvision.transforms import functional as F
from torchvision.ops import nms, box_convert
from torchvision.utils import draw_bounding_boxes
from typing import Annotated

# Load the model
rf_detr = RFDETRBase()
model = rf_detr.model.model.cpu().eval()
input_size: int = rf_detr.model.resolution

# Prepare for export
model.export()

class Detection(BaseModel):
    x_min: float = Field(description="Normalized minimum X coordinate.")
    y_min: float = Field(description="Normalized minimum Y coordinate.")
    x_max: float = Field(description="Normalized maximum X coordinate.")
    y_max: float = Field(description="Normalized maximum Y coordinate.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("rfdetr")
        .run_commands("uv pip uninstall -y opencv-python opencv-python-headless")
        .pip_install("opencv-python-headless"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, input_size, input_size)],
            exporter="none" # https://github.com/roboflow/rf-detr/issues/406
        )
    ]
)
@inference_mode()
def rf_detr(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.5,
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
    Detect objects in an image with RF-DETR.
    """
    # Preprocess image
    image = F.resize(image, [input_size, input_size])
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(image_tensor, RFDETRBase.means, RFDETRBase.stds)
    # Run model
    outputs = model(image_tensor[None])
    boxes_cxcywh: Tensor = outputs[0]
    logits: Tensor = outputs[1]
    # Compute scores
    scores = logits.sigmoid().view(logits.shape[0], -1)
    confidence_mask = scores >= min_confidence
    # Filter and decode
    flat_indices = confidence_mask.nonzero(as_tuple=True)[1]
    box_indices = flat_indices // logits.shape[2]
    boxes = boxes_cxcywh[0, box_indices]
    scores = scores[confidence_mask]
    labels = flat_indices % logits.shape[2]
    # Apply NMS
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    keep_indices = nms(boxes_xyxy, scores=scores, iou_threshold=max_iou)
    boxes = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    # Create detections
    detections = [Detection(
        x_min=box[0].item(),
        y_min=box[1].item(),
        x_max=box[2].item(),
        y_max=box[3].item(),
        label=COCO_CLASSES[label.item()],
        confidence=score.item(),
    ) for box, score, label in zip(boxes, scores, labels)]
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
    boxes = tensor([[det.x_min, det.y_min, det.x_max, det.y_max] for det in detections])
    boxes *= tensor(image.size).repeat(2)
    labels = [detection.label for detection in detections]
    result_tensor = draw_bounding_boxes(
        image_tensor,
        boxes=boxes,
        labels=labels,
        width=8,
        font="Arial.ttf",
        font_size=int(0.04 * image.width)
    )
    return F.to_pil_image(result_tensor)

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect objects
    image_path = Path(__file__).parent / "demo" / "vehicles.jpg"
    image = Image.open(image_path)
    detections = rf_detr(image)
    # Visualize
    print_json(data=[det.model_dump() for det in detections])
    _visualize_detections(image, detections).show()