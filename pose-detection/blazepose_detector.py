#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "opencv-python-headless", "tensorflow", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import TFLiteInterpreterMetadata
from numpy import array, ceil, ndarray, sqrt
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from shutil import unpack_archive
from tensorflow import lite
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
    rotation: float = Field(description="Bounding box rect.")

class PoseDetection(BaseModel):
    face: Rect = Field(description="Face rectangle.")
    body: RotatedRect = Field(description="Rotated body rectangle.")
    keypoints: list[tuple[float, float]] = Field(description="Upper-body center, upper-body rotation point, full-body center, full-body rotation point.")
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
    Generates SSD anchors matching the BlazePose detector configuration.
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
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and (strides[last_same_stride_layer] == strides[layer_id]):
            scale = calculate_scale(min_scale, max_scale, last_same_stride_layer, strides_size)
            if last_same_stride_layer == 0 and reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                layer_aspect_ratios.append(1.0)
                layer_aspect_ratios.append(2.0)
                layer_aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
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

# Download landmark model (there are 3 variants)
# Options: pose_landmarker_lite, pose_landmarker_full, pose_landmarker_heavy
MODEL_VARIANT = "pose_landmarker_lite"  # Change this to switch models
model_task_url = (
    f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"{MODEL_VARIANT}/float16/latest/{MODEL_VARIANT}.task"
)
model_path = Path("pose_detector.tflite")
if not model_path.exists():
    model_task_path = Path(Path(model_task_url).name)
    print(f"Downloading {MODEL_VARIANT}...")
    download_url_to_file(model_task_url, model_task_path)
    unpack_archive(model_task_path, format="zip")

# Load the detector model
interpreter = lite.Interpreter("pose_detector.tflite")
interpreter.allocate_tensors()

# Get detector I/O tensor indices
_get_tensor_idx = lambda details, name: next(detail for detail in details if detail["name"] == name)["index"]
interpreter_image_idx = _get_tensor_idx(interpreter.get_input_details(), "input_1")
interpreter_logit_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity")
interpreter_score_idx = _get_tensor_idx(interpreter.get_output_details(), "Identity_1")

# Generate anchors for detection
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt#L84-L105
_, input_height, input_width, _ = interpreter.get_tensor(interpreter_image_idx).shape
ANCHORS = _generate_ssd_anchors(**{
    "num_layers": 5,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_width": input_width,
    "input_size_height": input_height,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "fixed_anchor_size": True
})

@compile(
    tag="@mediapipe/blazepose-detector-lite",
    description="Detect pose ROIs in an image with BlazePose Detector (lite).",
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("tensorflow"),
    access="public",
    metadata=[
        TFLiteInterpreterMetadata(interpreter=interpreter, model_path=model_path),
    ]
)
def detect_poses(
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
) -> Annotated[list[PoseDetection], Parameter.Generic(description="Detected pose ROIs.")]:
    """
    Infer pose detections from an image using BlazePose Detector.
    """
    # Preprocess image
    image_tensor, scale_factors = _preprocess_image(image, input_size=224)
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
    # Create boxes
    raw_boxes = logits[:,:4]
    raw_keypoints = logits[:,4:].view(-1, 4, 2) 
    x_center = raw_boxes[:,0] / input_width + anchors[:,0]
    y_center = raw_boxes[:,1] / input_height + anchors[:,1]
    widths = raw_boxes[:,2] / input_width
    heights = raw_boxes[:,3] / input_height
    boxes_cxcywh = stack([x_center, y_center, widths, heights], dim=1) * scale_factors.repeat(2)
    # Create keypoints
    kp_x = raw_keypoints[:,:,0] / input_width + anchors[:,None,0]
    kp_y = raw_keypoints[:,:,1] / input_height + anchors[:,None,1]
    keypoints = stack((kp_x, kp_y), dim=-1) * scale_factors
    # Apply NMS
    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
    keep_indices = nms(boxes_xyxy, scores=scores, iou_threshold=max_iou)
    boxes_cxcywh = boxes_cxcywh[keep_indices]
    scores = scores[keep_indices]
    keypoints = keypoints[keep_indices]
    # Create detections
    detections = [_create_pose_detection(
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
    fill: int=114
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
    image_tensor = image_tensor * 2. - 1.
    scale_factors = tensor([scaled_width / input_size, scaled_height / input_size]).reciprocal()
    # Return
    return image_tensor, scale_factors

def _create_pose_detection(
    *,
    box: Tensor, # cxcywh
    score: Tensor,
    keypoints: Tensor,
    scale_factors: Tensor
) -> PoseDetection:
    """
    Create a pose detection from BlazePose Detector.
    """
    # Create face
    face = Rect(
        x_center=box[0].item(),
        y_center=box[1].item(),
        width=box[2].item(),
        height=box[3].item()
    )
    # Calculate body rotation and size
    rotation_vec = keypoints[1] - keypoints[0]
    angle = rad2deg(arctan2(rotation_vec[0], -rotation_vec[1]))
    distance = 2. * rotation_vec.norm()
    size = tensor([distance, distance]) * scale_factors
    # Extract body rectangle coordinates
    body = RotatedRect(
        x_center=keypoints[0,0].item(),
        y_center=keypoints[0,1].item(),
        width=size[0].item(),
        height=size[1].item(),
        rotation=angle.item()
    )
    detection = PoseDetection(
        face=face,
        body=body,
        keypoints=keypoints.tolist(),
        confidence=score.item()
    )
    # Return
    return detection

def _visualize_pose_detections(
    image: Image.Image,
    detections: list[PoseDetection]
) -> Image.Image:
    """
    Visualize detection results on an image.
    """
    from cv2 import (
        boxPoints, circle, cvtColor, polylines, putText, rectangle,
        COLOR_BGR2RGB, COLOR_RGB2BGR, FONT_HERSHEY_SIMPLEX, LINE_AA
    )
    image_rgb = image.convert("RGB")
    canvas = cvtColor(array(image_rgb), COLOR_RGB2BGR)
    height, width, _ = canvas.shape
    for detection in detections:
        # Draw face rectangle (axis-aligned)
        face = detection.face
        face_xmin = int((face.x_center - face.width / 2.0) * width)
        face_ymin = int((face.y_center - face.height / 2.0) * height)
        face_xmax = int((face.x_center + face.width / 2.0) * width)
        face_ymax = int((face.y_center + face.height / 2.0) * height)
        pt1, pt2 = (face_xmin, face_ymin), (face_xmax, face_ymax),
        rectangle(canvas, pt1, pt2, color=(0, 255, 0), thickness=2)
        # Draw rotated body rectangle
        body = detection.body
        center = (body.x_center * width, body.y_center * height)
        size = (max(body.width * width, 1.0), max(body.height * height, 1.0))
        rect = (center, size, body.rotation)
        box_points = boxPoints(rect).astype(int)
        polylines(canvas, [box_points], True, (0, 128, 255), 2)
        # Optionally visualize keypoints
        for idx, kp in enumerate(detection.keypoints):
            x, y = int(kp[0] * width), int(kp[1] * height)
            circle(canvas, (x, y), radius=10, color=(255, 0, 0), thickness=-1)
            putText(canvas, f"{idx}", (x + 4, y - 4), FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, LINE_AA)
    result_rgb = cvtColor(canvas, COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print_json    
    # Detect poses
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    poses = detect_poses(image, min_confidence=0.5, max_iou=0.1)
    # Visualize
    print_json(data=TypeAdapter(list[PoseDetection]).dump_python(poses))
    annotated_image = _visualize_pose_detections(image, poses)
    annotated_image.show()