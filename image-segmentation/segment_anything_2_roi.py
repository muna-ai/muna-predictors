#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import array, bool_, zeros
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from torch import from_numpy
from torch.nn.functional import interpolate
from torchvision.transforms.functional import normalize, resize, to_tensor
from typing import Annotated

# Download SAM 2.1 models
REPO_ID = "vietanhdev/segment-anything-2-onnx-models"
encoder_path = hf_hub_download(repo_id=REPO_ID, filename="sam2_hiera_small.encoder.onnx")
decoder_path = hf_hub_download(repo_id=REPO_ID, filename="sam2_hiera_small.decoder.onnx")

# Create inference sessions
encoder_session = InferenceSession(encoder_path)
decoder_session = InferenceSession(decoder_path)

class ROI(BaseModel):
    x_min: float = Field(description="Normalized minimum X coordinate.")
    y_min: float = Field(description="Normalized minimum Y coordinate.")
    x_max: float = Field(description="Normalized maximum X coordinate.")
    y_max: float = Field(description="Normalized maximum Y coordinate.")

@compile(
    tag="@meta/segment-anything-2-roi",
    description="Segment an image region of interest using Segment Anything Model 2.",
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=encoder_session, model_path=encoder_path),
        OnnxRuntimeInferenceSessionMetadata(session=decoder_session, model_path=decoder_path)
    ]
)
def predict(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")],
    roi: Annotated[ROI, Parameter.BoundingBox(description="Region of interest in normalized coordinates")],
    mask_threshold: Annotated[float, Parameter.Numeric(
        description="Mask confidence threshold.",
        min=0.,
        max=1.
    )]=0.1
) -> Annotated[
    NDArray[bool_],
    Parameter.Generic(description="Boolean mask with shape (H,W).")
]:
    """
    Segment an image region of interest using Segment Anything Model 2.
    """
    # Preprocess image
    width, height = image.size
    image = image.convert("RGB")
    image = resize(image, (1024, 1024))
    image_tensor = to_tensor(image)
    image_tensor = normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_tensor = image_tensor[None]
    # Compute image embeddings
    high_res_feats_0, high_res_feats_1, image_embeddings = encoder_session.run(None, {
        "image": image_tensor.numpy()
    })
    # Decode masks
    num_points = 1
    point_coords = array([
        [roi.x_min, roi.y_min],
        [roi.x_max, roi.y_max]
    ], dtype="float32")
    point_coords[...,0] = point_coords[...,0] * 1024
    point_coords[...,1] = point_coords[...,1] * 1024
    point_coords = point_coords[None] # (N,2,2)
    point_labels = array([2, 3], dtype="float32")[None] # (N,2)
    mask_input = zeros((num_points, 1, 256, 256), dtype="float32")
    has_mask_input = zeros((1,), dtype="float32")
    masks, iou_predictions = decoder_session.run(None, {
        "image_embed": image_embeddings,
        "high_res_feats_0": high_res_feats_0,
        "high_res_feats_1": high_res_feats_1,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input
    })
    # Extract best mask
    mask_idx = iou_predictions.argmax()
    mask = masks[:,mask_idx:mask_idx+1,...]
    mask = interpolate(
        from_numpy(mask),
        size=(height, width),
        mode="bilinear",
        align_corners=False
    ).numpy()
    mask = mask.squeeze()
    mask = mask > mask_threshold
    # Return
    return mask

if __name__ == "__main__":
    # Predict
    image_path = Path(__file__).parent / "demo" / "fruits.jpg"
    image = Image.open(image_path)
    roi = ROI(x_min=0.08, y_min=0.51, x_max=0.34, y_max=0.74)
    mask = predict(image, roi)
    # Show mask
    mask_image = Image.fromarray(mask)
    mask_image.show()