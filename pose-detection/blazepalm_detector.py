#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["ai-edge-litert", "muna", "opencv-python-headless", "torchvision"]
# ///

from ai_edge_litert.interpreter import Interpreter
from muna import compile, Parameter, Sandbox
from muna.beta import TFLiteInterpreterMetadata
from numpy import array, ceil, ndarray, sqrt
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from shutil import unpack_archive
from torch import arctan2, from_numpy, rad2deg, stack, tensor, Tensor
from torch.hub import download_url_to_file
from torchvision.ops import nms, box_convert
from torchvision.transforms import functional as F
from typing import Annotated

class Rect(BaseModel):
    x_center: float = Field(description="Normalized bounding box X-coordinate.")
    y_center: float = Field(description="Normalized bounding box Y-coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")

class RotatedRect(Rect):
    rotation: float = Field(description="Bounding box rotation in degrees.")

class HandDetection(BaseModel):
    palm: Rect = Field(description="Palm bounding box.")
    roi: RotatedRect = Field(description="Rotated hand region of interest for landmark detection.")
    keypoints: list[tuple[float, float]] = Field(description="Palm keypoints: wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp, thumb_cmc, thumb_tip.")
    confidence: float = Field(description="Detection confidence score.")

def _generate_ssd_anchors(
    *,
    input_size_height: int,
    input_size_width: int,
    min_scale: float,
    max_scale: float,
    strides: list[int],
    num_layers: int,
    aspect_ratios: list[float],
    interpolated_scale_aspect_ratio: float=1.0,
    anchor_offset_x: float=0.5,
    anchor_offset_y: float=0.5,
    reduce_boxes_in_lowest_layer: bool=False,
    fixed_anchor_size: bool=False
) -> ndarray:
    """
    Generates SSD anchors matching the palm detector configuration.
    """
    def calculate_scale(
        min_scale_value: float,
        max_scale_value: float,
        stride_index: int,
        num_strides: int,
    ) -> float:
        if num_strides == 1:
            return min_scale_value
        return min_scale_value + ((max_scale_value - min_scale_value) * stride_index / (num_strides - 1.0))
    strides_size = len(strides)
    assert num_layers == strides_size
    anchors: list[list[float]] = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_heights: list[float] = []
        anchor_widths: list[float] = []
        layer_aspect_ratios: list[float] = []
        scales: list[float] = []
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and (strides[last_same_stride_layer] == strides[layer_id]):
            scale = calculate_scale(min_scale, max_scale, last_same_stride_layer, strides_size)
            if last_same_stride_layer == 0 and reduce_boxes_in_lowest_layer:
                layer_aspect_ratios.extend([1.0, 2.0, 0.5])
                scales.extend([0.1, scale, scale])
            else:
                for aspect_ratio in aspect_ratios:
                    layer_aspect_ratios.append(aspect_ratio)
                    scales.append(scale)
                if interpolated_scale_aspect_ratio > 0.0:
                    scale_next = (
                        1.0
                        if last_same_stride_layer == strides_size - 1
                        else calculate_scale(min_scale, max_scale, last_same_stride_layer + 1, strides_size)
                    )
                    scales.append(sqrt(scale * scale_next))
                    layer_aspect_ratios.append(interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        for i in range(len(layer_aspect_ratios)):
            ratio_sqrts = sqrt(layer_aspect_ratios[i])
            anchor_heights.append(scales[i] / ratio_sqrts)
            anchor_widths.append(scales[i] * ratio_sqrts)
        stride = strides[layer_id]
        feature_map_height = int(ceil(input_size_height / stride))
        feature_map_width = int(ceil(input_size_width / stride))
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_heights)):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    y_center = (y + anchor_offset_y) / feature_map_height
                    new_anchor = [x_center, y_center, 0, 0]
                    if fixed_anchor_size:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_widths[anchor_id]
                        new_anchor[3] = anchor_heights[anchor_id]
                    anchors.append(new_anchor)
        layer_id = last_same_stride_layer
    return array(anchors, dtype="float32")

# Download hand landmarker model
model_task_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
detector_model_path = Path("hand_detector.tflite")
if not detector_model_path.exists():
    model_task_path = Path("hand_landmarker.task")
    print("Downloading hand_landmarker.task...")
    download_url_to_file(model_task_url, model_task_path)
    unpack_archive(model_task_path, format="zip")

# Load the detector model
interpreter = Interpreter(str(detector_model_path))
interpreter.allocate_tensors()

# Get detector I/O tensor indices
_get_tensor_idx = lambda details, name: next(detail for detail in details if detail["name"] == name)["index"]
interpreter_image_idx = _get_tensor_idx(interpreter.get_input_details(), "input_1")
interpreter_logit_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity")
interpreter_score_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_1")

# Generate anchors for palm detection
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
_, input_height, input_width, _ = interpreter.get_tensor(interpreter_image_idx).shape
ANCHORS = _generate_ssd_anchors(**{
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_width": input_width,
    "input_size_height": input_height,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "fixed_anchor_size": True
})

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ai-edge-litert"),
    metadata=[
        TFLiteInterpreterMetadata(
            interpreter=interpreter,
            model_path=detector_model_path
        ),
    ]
)
def blazepalm_detector(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.5,
    max_iou: Annotated[float, Parameter.Numeric(
        description="Maximum intersection-over-union score to discard overlapping detections.",
        min=0.,
        max=1.
    )]=0.3
) -> Annotated[
    list[HandDetection],
    Parameter.BoundingBoxes(description="Detected hand ROIs.")
]:
    """
    Detect hands in an image using the MediaPipe palm detector.
    """
    # Preprocess image
    image_tensor, scale_factors = _preprocess_image(image, input_size=192)
    # Run detector model
    interpreter.set_tensor(interpreter_image_idx, image_tensor[None].numpy())
    interpreter.invoke()
    logits = from_numpy(interpreter.get_tensor(interpreter_logit_idx))[0]
    scores = from_numpy(interpreter.get_tensor(interpreter_score_idx))[0].sigmoid().squeeze()
    # Filter by confidence
    confidence_mask = scores > min_confidence
    logits = logits[confidence_mask]
    scores = scores[confidence_mask]
    anchors = ANCHORS[confidence_mask]
    # Decode boxes: first 4 values are (cx_offset, cy_offset, w, h)
    raw_boxes = logits[:, :4]
    raw_keypoints = logits[:, 4:].view(-1, 7, 2)
    x_center = raw_boxes[:, 0] / input_width + anchors[:, 0]
    y_center = raw_boxes[:, 1] / input_height + anchors[:, 1]
    widths = raw_boxes[:, 2] / input_width
    heights = raw_boxes[:, 3] / input_height
    boxes_cxcywh = stack([x_center, y_center, widths, heights], dim=1) * scale_factors.repeat(2)
    # Decode keypoints
    kp_x = raw_keypoints[:, :, 0] / input_width + anchors[:, None, 0]
    kp_y = raw_keypoints[:, :, 1] / input_height + anchors[:, None, 1]
    keypoints = stack([kp_x, kp_y], dim=-1) * scale_factors
    # Apply NMS
    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
    keep_indices = nms(boxes_xyxy, scores=scores, iou_threshold=max_iou)
    boxes_cxcywh = boxes_cxcywh[keep_indices]
    scores = scores[keep_indices]
    keypoints = keypoints[keep_indices]
    # Create detections
    detections = [_create_hand_detection(
        box=box,
        score=score,
        keypoints=kp,
        scale_factors=scale_factors
    ) for box, score, kp in zip(boxes_cxcywh, scores, keypoints)]
    # Return
    return detections

def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int,
    fill: int=128
) -> tuple[Tensor, Tensor]:
    """
    Preprocess an image for inference by downscaling and padding it to have a square aspect.
    """
    # Compute scaled size and padding
    image_width, image_height = image.size
    ratio = min(input_size / image_width, input_size / image_height)
    scaled_width = int(image_width * ratio)
    scaled_height = int(image_height * ratio)
    padding = [0, 0, input_size - scaled_width, input_size - scaled_height]
    # Downscale and pad image
    image = F.resize(image, [scaled_height, scaled_width])
    image = image.convert("RGB")
    image = F.pad(image, padding, fill=fill)
    # Create tensors
    image_tensor = from_numpy(array(image, dtype="float32")) / 255.0
    scale_factors = tensor([scaled_width / input_size, scaled_height / input_size]).reciprocal()
    # Return
    return image_tensor, scale_factors

def _create_hand_detection(
    *,
    box: Tensor,
    score: Tensor,
    keypoints: Tensor,
    scale_factors: Tensor
) -> HandDetection:
    """
    Create a hand detection with a rotated ROI for landmark detection.
    """
    # Create palm rect
    palm = Rect(
        x_center=box[0].item(),
        y_center=box[1].item(),
        width=box[2].item(),
        height=box[3].item()
    )
    # Calculate hand ROI rotation from wrist (kp0) to middle_mcp (kp2)
    wrist = keypoints[0]
    middle_mcp = keypoints[2]
    rotation_vec = middle_mcp - wrist
    angle = rad2deg(arctan2(rotation_vec[0], -rotation_vec[1]))
    # Compute ROI size: expand to cover the full hand
    distance = 2.5 * rotation_vec.norm()
    size = tensor([distance, distance]) * scale_factors
    # ROI center shifted toward fingers from the wrist
    roi_center_x = (wrist[0] + middle_mcp[0]) * 0.5
    roi_center_y = (wrist[1] + middle_mcp[1]) * 0.5
    roi = RotatedRect(
        x_center=roi_center_x.item(),
        y_center=roi_center_y.item(),
        width=size[0].item(),
        height=size[1].item(),
        rotation=angle.item()
    )
    detection = HandDetection(
        palm=palm,
        roi=roi,
        keypoints=keypoints.tolist(),
        confidence=score.item()
    )
    # Return
    return detection

def _visualize_hand_detections(
    image: Image.Image,
    detections: list[HandDetection]
) -> Image.Image:
    """
    Visualize hand detection results on an image.
    """
    from cv2 import (
        boxPoints, circle, cvtColor, polylines, putText, rectangle,
        COLOR_BGR2RGB, COLOR_RGB2BGR, FONT_HERSHEY_SIMPLEX, LINE_AA
    )
    image_rgb = image.convert("RGB")
    canvas = cvtColor(array(image_rgb), COLOR_RGB2BGR)
    height, width, _ = canvas.shape
    for detection in detections:
        # Draw palm rectangle
        palm = detection.palm
        palm_xmin = int((palm.x_center - palm.width / 2.0) * width)
        palm_ymin = int((palm.y_center - palm.height / 2.0) * height)
        palm_xmax = int((palm.x_center + palm.width / 2.0) * width)
        palm_ymax = int((palm.y_center + palm.height / 2.0) * height)
        rectangle(canvas, (palm_xmin, palm_ymin), (palm_xmax, palm_ymax), color=(0, 255, 0), thickness=2)
        # Draw rotated ROI
        roi = detection.roi
        center = (roi.x_center * width, roi.y_center * height)
        size = (max(roi.width * width, 1.0), max(roi.height * height, 1.0))
        rect = (center, size, roi.rotation)
        box_points = boxPoints(rect).astype(int)
        polylines(canvas, [box_points], True, (0, 128, 255), 2)
        # Draw keypoints
        for idx, kp in enumerate(detection.keypoints):
            x, y = int(kp[0] * width), int(kp[1] * height)
            circle(canvas, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            putText(canvas, f"{idx}", (x + 4, y - 4), FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, LINE_AA)
        # Add score text
        score_text = f"{detection.confidence:.2f}"
        putText(canvas, score_text, (palm_xmin, palm_ymin - 5), FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, LINE_AA)
    result_rgb = cvtColor(canvas, COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print_json
    # Detect hands
    image_path = Path(__file__).parent / "demo" / "palms.jpg"
    image = Image.open(image_path)
    hands = blazepalm_detector(image, min_confidence=0.5, max_iou=0.3)
    # Visualize
    print_json(data=TypeAdapter(list[HandDetection]).dump_python(hands))
    annotated_image = _visualize_hand_detections(image, hands)
    annotated_image.save("hand_detection_result.jpg")
    print(f"Detected {len(hands)} hand(s)")
