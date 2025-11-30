#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["muna", "rich", "torchvision", "ultralytics"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from numpy import bool_
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field
from torch import empty, from_numpy, inference_mode, randn, tensor, Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from torchvision.ops import batched_nms, box_convert
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from typing import Annotated
from ultralytics import YOLO

class Detection(BaseModel):    
    x_center: float = Field(description="Normalized bounding box center X-coordinate.")
    y_center: float = Field(description="Normalized bounding box center Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

class YOLOSegWrapper(Module):
    """
    This is a wrapper around the YOLO segmentation model.
    We do this because the original YOLO model returns a nested tuple.
    But inference backends expect a flat list of output tensors.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(input)
        logits: Tensor = outputs[0]
        mask_prototypes: Tensor = outputs[1][-1]
        return logits, mask_prototypes

# Create the model
yolo = YOLO("yolov8l-seg.pt")
model = YOLOSegWrapper(yolo.model).eval()
labels: dict[int, str] = yolo.model.names

# Dry run the model for export
INPUT_SIZE = 640 # pixels
model_args = [randn(1, 3, INPUT_SIZE, INPUT_SIZE)]
model(*model_args)

@compile(
    tag="@ultralytics/yolo-v8-segment-large",
    description="Segment objects in an image with YOLO-v8 (large).",
    access="private", # YOLO-v8 is under AGPL license
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ultralytics")
        .pip_install("opencv-python-headless"),
    metadata=[
        OnnxRuntimeInferenceMetadata(model=model, model_args=model_args)
    ]
)
@inference_mode()
def segment_image(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")],
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
    )]=0.45
) -> tuple[
    Annotated[list[Detection], Parameter.Generic(description="Detected objects.")],
    Annotated[NDArray[bool_], Parameter.Generic(description="Segmentation masks with shape (M,H,W).")]
]:
    """
    Segment objects in an image with YOLO-v8 (large).
    """
    # Preprocess
    image_tensor, scale_factors = _preprocess_image(image, input_size=INPUT_SIZE)
    # Run model
    model_outputs = model(image_tensor[None])
    logits = model_outputs[0]                       # (1,4+C+M,8400)
    mask_prototypes = model_outputs[1]              # (1,M,mH,mW)
    # Parse detection predictions
    num_classes = len(labels)                       # C
    mask_idx = 4 + num_classes
    predictions = logits[0].T                       # (8400,4+C+M)
    boxes_cxcywh = predictions[:,:4]                # (8400,4)
    class_scores = predictions[:,4:mask_idx]        # (8400,C)
    mask_coefficients = predictions[:,mask_idx:]    # (8400,M)
    max_scores, class_ids = class_scores.max(dim=1) # (8400,), (8400,)
    # Filter by confidence
    confidence_mask = max_scores >= min_confidence
    filtered_boxes = boxes_cxcywh[confidence_mask] * scale_factors
    filtered_scores = max_scores[confidence_mask]
    filtered_class_ids = class_ids[confidence_mask]
    filtered_mask_coeffs = mask_coefficients[confidence_mask]
    # Check if any detections remain
    if len(filtered_boxes) == 0:
        return []
    # Apply NMS
    filtered_boxes_xyxy = box_convert(
        filtered_boxes,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    )
    keep_indices = batched_nms(
        filtered_boxes_xyxy,
        scores=filtered_scores,
        idxs=filtered_class_ids,
        iou_threshold=max_iou
    )
    # Select final detections
    final_boxes = filtered_boxes[keep_indices]
    final_scores = filtered_scores[keep_indices]
    final_class_ids = filtered_class_ids[keep_indices]
    final_mask_coeffs = filtered_mask_coeffs[keep_indices]
    final_boxes_xyxy = filtered_boxes_xyxy[keep_indices]
    # Create detection objects
    detections = [_create_detection(
        box,
        class_id=class_id,
        score=score
    ) for box, class_id, score in zip(final_boxes, final_class_ids, final_scores)]
    # Generate masks as stacked numpy array (N,H,W)
    masks = _generate_masks(
        mask_prototypes[0],
        final_mask_coeffs,
        final_boxes_xyxy,
        image.size,
        scale_factors
    )
    # Return
    return detections, masks

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
    image = F.resize(image, [scaled_height, scaled_width])
    image = image.convert("RGB")
    image = F.pad(image, image_padding, fill=114)
    # Create tensors
    image_tensor = F.to_tensor(image)
    scaled_sizes = tensor([scaled_width, scaled_height, scaled_width, scaled_height])
    # Return
    return image_tensor, scaled_sizes.reciprocal()

def _generate_masks(
    mask_prototypes: Tensor,
    mask_coefficients: Tensor,
    boxes_xyxy: Tensor, 
    image_size: tuple[int, int],
    scale_factors: Tensor
) -> NDArray[bool_]:
    """
    Generate segmentation masks from prototypes and coefficients.
    """
    # Check empty
    image_width, image_height = image_size
    if len(mask_coefficients) == 0:
        return empty(0, image_height, image_width).bool()
    # Generate masks through matrix multiplication
    c, mh, mw = mask_prototypes.shape
    masks = (mask_coefficients @ mask_prototypes.view(c, -1)).view(-1, mh, mw) # (2,160,160)
    # Crop the mask to account for padding on the input image
    scaled_size: list[int] = scale_factors.reciprocal().int()
    crop_width = int(mw * scaled_size[0].item() / INPUT_SIZE)
    crop_height = int(mh * scaled_size[1].item() / INPUT_SIZE)
    crop_masks = masks[:,:crop_height,:crop_width]
    # Upsample masks to original image size
    upsampled_masks = interpolate(
        crop_masks[None],
        size=(image_height, image_width),
        scale_factor=None,
        mode="bilinear"
    )[0]
    # Crop masks to bounding boxes
    boxes_pixel = boxes_xyxy * tensor([image_width, image_height, image_width, image_height])
    result_masks = _crop_masks(upsampled_masks, boxes_pixel)
    # Binarize
    binary_masks = result_masks > 0.
    # Return
    return binary_masks.numpy()

def _crop_masks(masks: Tensor, boxes: Tensor) -> Tensor:
    """
    Crop masks to bounding box regions.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes.chunk(4, 1)      # each (N,1)
    # Create coordinate grids
    r = tensor(range(w))[None,None,:]       # (1, 1, w)
    c = tensor(range(h))[None,:,None]       # (1, h, 1)
    # Expand coordinate grids to (N,H,W)
    r = r.expand(n, h, w)
    c = c.expand(n, h, w)
    # Expand box coordinates to (N,H,W)
    x1 = x1[:,:,None].expand(-1, h, w)
    x2 = x2[:,:,None].expand(-1, h, w)
    y1 = y1[:,:,None].expand(-1, h, w)
    y2 = y2[:,:,None].expand(-1, h, w)
    # Create mask for valid regions
    crop_mask = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    # Mask
    return masks * crop_mask

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
    detections: list[Detection],
    masks: NDArray[bool_],
    *,
    alpha: float=0.6
) -> Image.Image:
    """
    Visualize detections.
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
    result_tensor = draw_segmentation_masks(
        result_tensor,
        from_numpy(masks).bool(),
        alpha=alpha
    )
    return F.to_pil_image(result_tensor)

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Segment objects
    image_path = Path(__file__).parent / "demo" / "fruits.jpg"
    image = Image.open(image_path)
    detections, masks = segment_image(image)
    # Print detections
    print_json(data=[det.model_dump() for det in detections])
    # Visualize results
    annotated_image = _visualize_detections(image, detections, masks)
    annotated_image.show()