#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub", "muna", "onnxruntime",
#   "opencv-python-headless", "torchvision"
# ]
# ///

from cv2 import (
    boundingRect, boxPoints, dilate, fillPoly, findContours, getPerspectiveTransform,
    getStructuringElement, minAreaRect, warpPerspective, CHAIN_APPROX_SIMPLE,
    MORPH_RECT, RETR_LIST
)
from huggingface_hub import hf_hub_download
from pathlib import Path
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import (
    argmax, array, clip, concatenate, float32, int32,
    max as np_max, mean, ndarray, sum, transpose,
    uint8, zeros
)
from onnxruntime import InferenceSession
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms import functional as F
from typing import Annotated

# Download ONNX models
REPO_ID = "SWHL/RapidOCR"
detector_model_path = hf_hub_download(repo_id=REPO_ID, filename="PP-OCRv4/ch_PP-OCRv4_det_infer.onnx")
classifier_model_path = hf_hub_download(repo_id=REPO_ID, filename="PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx")
recognizer_model_path = hf_hub_download(repo_id=REPO_ID, filename="PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx")

# Download character dictionary
dict_path = hf_hub_download(repo_id="karmueo/PaddleOcr", filename="ppocr_keys_v1.txt")

# Load ONNX sessions
detector_session = InferenceSession(detector_model_path)
classifier_session = InferenceSession(classifier_model_path)
recognizer_session = InferenceSession(recognizer_model_path)

# Load character dictionary
CHARACTER_DICT = (
    ["blank"] +
    [l.strip() for l in Path(dict_path).read_text().splitlines()] +
    [" "]
)

# OCR result
class OcrResult(BaseModel):
    text: str = Field(description="Recognized text.")
    confidence: float = Field(description="Recognition confidence score.")
    box: list[float] = Field(description="Normalized bounding box as (x,y,w,h).")

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "onnxruntime", "opencv-python-headless"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=detector_session, model_path=detector_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=classifier_session, model_path=classifier_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=recognizer_session, model_path=recognizer_model_path),
    ]
)
def rapid_ocr(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image for OCR.")
    ],
    *,
    text_score: Annotated[float, Parameter.Numeric(
        description="Minimum confidence threshold for recognized text.",
        min=0.,
        max=1.
    )]=0.5,
    min_detection_score: Annotated[float, Parameter.Numeric(
        description="Minimum score for a region to be detected as text.",
        min=0.,
        max=1.
    )]=0.3,
    min_orientation_score: Annotated[float, Parameter.Numeric(
        description="Minimum score to apply orientation correction to a text region.",
        min=0.,
        max=1.
    )]=0.9,
    min_box_score: Annotated[float, Parameter.Numeric(
        description="Minimum score for a detected box to be kept.",
        min=0.,
        max=1.
    )]=0.5,
    max_detection_size: Annotated[int, Parameter.Numeric(
        description="Maximum side length for the detection input image.",
        min=32
    )]=736,
    box_expansion_ratio: Annotated[float, Parameter.Numeric(
        description="Ratio to expand detected text boxes.",
        min=0.
    )]=1.6,
    max_candidates: Annotated[int, Parameter.Numeric(
        description="Maximum number of text region candidates to process.",
        min=1,
        max=2_000
    )]=1000,
) -> Annotated[
    list[OcrResult],
    Parameter.BoundingBoxes(description="OCR results with text, confidence, and bounding boxes.")
]:
    """
    Recognize text in an image using RapidOCR (PP-OCRv4).
    """
    # Detect text ROIs
    image = image.convert("RGB")
    det_boxes = _detect_text_rois(
        image,
        threshold=min_detection_score,
        min_box_score=min_box_score,
        max_size=max_detection_size,
        expansion_ratio=box_expansion_ratio,
        max_candidates=max_candidates,
    )
    if len(det_boxes) == 0:
        return []
    # Crop text regions
    crops = [_get_image_roi(image, box) for box in det_boxes]
    # Make crop regions upright
    upright_crops = [_correct_text_orientation(
        crop,
        threshold=min_orientation_score
    ) for crop in crops]
    # Recognize text regions
    results = [_make_ocr_result(crop, box) for crop, box in zip(upright_crops, det_boxes)]
    results = [r for r in results if r.confidence >= text_score and len(r.text) > 0]
    # Return
    return results

def _detect_text_rois(
    image: Image.Image,
    *,
    threshold: float,
    min_box_score: float,
    max_size: int,
    expansion_ratio: float,
    max_candidates: int,
) -> ndarray:
    """
    Run text detection and return bounding boxes.
    """
    orig_w, orig_h = image.size
    target_w, target_h = _get_detector_image_size(orig_w, orig_h, max_size=max_size)
    ratio_w = float(target_w) / float(orig_w)
    ratio_h = float(target_h) / float(orig_h)
    resized = F.resize(image, (target_h, target_w))
    input_tensor = F.normalize(
        F.to_tensor(resized),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run detection model
    outputs = detector_session.run(None, {
        "x": input_tensor[None].numpy()
    })
    pred = outputs[0][0, 0]  # (H, W) probability map
    # Post-process: threshold and find contours
    segmentation = (pred > threshold).astype(uint8)
    kernel = getStructuringElement(MORPH_RECT, (2, 2))
    segmentation = dilate(segmentation, kernel)
    contours, _ = findContours(segmentation, RETR_LIST, CHAIN_APPROX_SIMPLE)
    # Process contours into boxes
    n_contours = min(len(contours), max_candidates)
    out = zeros((n_contours, 4, 2), dtype=float32)
    count = 0
    for i in range(n_contours):
        contour = contours[i]
        rect = minAreaRect(contour)
        box = boxPoints(rect)
        score = _box_score(pred, contour)
        if score < min_box_score:
            continue
        box = _unclip(box, expansion_ratio)
        if box.shape[0] == 0:
            continue
        box = _order_points(box)
        box[:,0] = clip(box[:, 0] / ratio_w, 0, orig_w)
        box[:,1] = clip(box[:, 1] / ratio_h, 0, orig_h)
        out[count] = box.astype(float32)
        count += 1
    # Return
    return out[:count]

def _get_image_roi(image: Image.Image, box: ndarray) -> ndarray:
    """
    Crop and perspective-correct a text region from the image.
    """
    img = array(image)
    points = box.astype(float32)
    w1 = float(((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2) ** 0.5)
    w2 = float(((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2) ** 0.5)
    target_w = max(int(max(w1, w2)), 1)
    h1 = float(((points[3][0] - points[0][0]) ** 2 + (points[3][1] - points[0][1]) ** 2) ** 0.5)
    h2 = float(((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2) ** 0.5)
    target_h = max(int(max(h1, h2)), 1)
    dst_points = array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h],
    ], dtype=float32)
    M = getPerspectiveTransform(points, dst_points)
    cropped = warpPerspective(img, M, (target_w, target_h))
    if target_h > target_w * 1.5:
        cropped = transpose(cropped, (1, 0, 2))[:, ::-1, :]
    return cropped

def _correct_text_orientation(crop: ndarray, *, threshold: float) -> ndarray:
    """
    Classify text orientation and rotate 180° if needed.
    """
    # Pre-process
    target_w, pad_w = _get_classifier_image_size(crop)
    image = Image.fromarray(crop)
    image = F.resize(image, (48, target_w))
    image_tensor = F.normalize(
        F.to_tensor(image),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    padded_image_tensor = F.pad(image_tensor, [0, 0, pad_w, 0])
    # Run classifier
    outputs = classifier_session.run(None, {
        "x": padded_image_tensor[None].numpy()
    })
    # Post-process
    probs = outputs[0][0]
    label_idx = int(argmax(probs))
    score = float(probs[label_idx])
    if label_idx == 1 and score > threshold:
        return crop[::-1, ::-1, :]
    # Return
    return crop

def _make_ocr_result(crop: ndarray, points: ndarray) -> OcrResult:
    """
    Recognize text in a crop and return an OcrResult with its bounding box.
    """
    text, confidence = _recognize_text(crop)
    x = float(points[:, 0].min())
    y = float(points[:, 1].min())
    w = float(points[:, 0].max()) - x
    h = float(points[:, 1].max()) - y
    return OcrResult(text=text, confidence=confidence, box=[x, y, w, h])

def _recognize_text(crop: ndarray) -> tuple:
    """Recognize text in a cropped image."""
    # Pre-process
    target_w, pad_w = _get_recognizer_image_size(crop)
    image = Image.fromarray(crop)
    image = F.resize(image, (48, target_w))
    image_tensor = F.normalize(
        F.to_tensor(image),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    image_tensor = F.pad(image_tensor, [0, 0, pad_w, 0])
    # Inference
    outputs = recognizer_session.run(None, {
        "x": image_tensor[None].numpy()
    })
    # Post-process
    preds = outputs[0][0]  # (seq_len, num_classes)
    text, confidence = _ctc_decode(preds)
    # Return
    return text, confidence

def _box_score(pred: ndarray, contour: ndarray) -> float:
    """
    Calculate the mean prediction score inside a contour.
    """
    h, w = pred.shape[0], pred.shape[1]
    x, y, bw, bh = boundingRect(contour)
    x_end = min(x + bw, w)
    y_end = min(y + bh, h)
    mask = zeros((y_end - y, x_end - x), dtype=uint8)
    shifts = array([x, y], dtype=int32)
    pts = contour.reshape(-1, 2) - shifts
    shifted_contour = pts.reshape(-1, 1, 2)
    fillPoly(mask, [shifted_contour], 1)
    roi = pred[y:y_end, x:x_end]
    float_mask = mask.astype(float32)
    mask_count = sum(float_mask)
    masked_sum = sum(roi * float_mask)
    score = float(masked_sum / mask_count) if float(mask_count) > 0.0 else 0.0
    return score

def _unclip(box: ndarray, unclip_ratio: float):
    """
    Expand a polygon outward using the shoelace formula for area/perimeter.
    """
    n = len(box)
    area = 0.0
    for i in range(n):
        next_i = (i + 1) % n
        area += float(box[i][0]) * float(box[next_i][1])
        area -= float(box[next_i][0]) * float(box[i][1])
    area = abs(area) / 2.0
    if area == 0:
        return zeros((0, 2), dtype=float32)
    perimeter = 0.0
    for i in range(n):
        next_i2 = (i + 1) % n
        dx = float(box[next_i2][0]) - float(box[i][0])
        dy = float(box[next_i2][1]) - float(box[i][1])
        perimeter += (dx * dx + dy * dy) ** 0.5
    if perimeter == 0:
        return zeros((0, 2), dtype=float32)
    distance = area * unclip_ratio / perimeter
    rect = minAreaRect(box.reshape(-1, 1, 2).astype(float32))
    center_x = float(rect[0][0])
    center_y = float(rect[0][1])
    rect_w = float(rect[1][0]) + distance * 2
    rect_h = float(rect[1][1]) + distance * 2
    angle = float(rect[2])
    expanded_rect = ((center_x, center_y), (rect_w, rect_h), angle)
    result = boxPoints(expanded_rect)
    return result

def _order_points(box: ndarray) -> ndarray:
    """
    Order box points clockwise starting from top-left.
    """
    sorted_idx = box[:, 1].argsort()
    top = box[sorted_idx[:2]]
    bottom = box[sorted_idx[2:]]
    top = top[top[:, 0].argsort()]
    bottom = bottom[bottom[:, 0].argsort()[::-1]]
    return concatenate([top, bottom], axis=0)

def _get_detector_image_size(
    width: int,
    height: int,
    *,
    max_size: int
) -> tuple[int, int]:
    """Get the target image size for the detector, keeping sides as multiples of 32."""
    ratio = 1.0
    if max(height, width) > max_size:
        ratio = float(max_size) / float(max(height, width))
    target_h = max(32, (int(height * ratio) + 31) // 32 * 32)
    target_w = max(32, (int(width * ratio) + 31) // 32 * 32)
    return target_w, target_h

def _get_classifier_image_size(crop: ndarray) -> tuple[int, int]:
    """
    Get the target width and right-padding for the classifier.
    """
    h, w = crop.shape[0], crop.shape[1]
    new_w = min(int(w * 48.0 / h), 192)
    return new_w, max(192 - new_w, 0)

def _get_recognizer_image_size(crop: ndarray) -> tuple[int, int]:
    """
    Get the target width and right-padding for the recognizer.
    """
    h, w = crop.shape[0], crop.shape[1]
    new_w = max(min(int(w * 48.0 / h), 320), 1)
    return new_w, max(320 - new_w, 0)

def _ctc_decode(preds: ndarray) -> tuple[str, float]:
    """
    CTC greedy decode: collapse repeats and remove blanks.
    """
    pred_indices = argmax(preds, axis=1)
    max_probs = np_max(preds, axis=1)
    text = ""
    confidence_scores = [0.0]
    prev_idx = -1
    for i in range(len(pred_indices)):
        idx = int(pred_indices[i])
        if idx != prev_idx and idx != 0:  # not duplicate and not blank
            if idx < len(CHARACTER_DICT):
                text = text + CHARACTER_DICT[idx]
                confidence_scores.append(float(max_probs[i]))
        prev_idx = idx
    confidence_scores = confidence_scores[1:]
    confidence = float(mean(array(confidence_scores))) if len(confidence_scores) > 0 else 0.0
    return text, confidence

if __name__ == "__main__":
    image = Image.open("test/media/ocr_receipt.png")
    results = rapid_ocr(image)
    print(f"Found {len(results)} text regions:")
    for r in results:
        print(f"  [{r.confidence:.3f}] {r.text}")