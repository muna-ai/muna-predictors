#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub",
#     "muna",
#     "onnxruntime",
#     "opencv-python-headless",
#     "torchvision"
# ]
# ///

from cv2 import applyColorMap, cvtColor, COLOR_BGR2RGB, COLORMAP_INFERNO
from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import ndarray, uint8
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from PIL import Image
from torchvision.transforms import functional as F
from typing import Annotated

# Download FP16 ONNX model from HuggingFace
model_id = "onnx-community/DepthPro-ONNX"
model_path = hf_hub_download(
    repo_id=model_id,
    subfolder="onnx",
    filename="model_fp16.onnx"
)

# Load ONNX model
# Disable graph optimizations for FP16 (SimplifiedLayerNormFusion bug)
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
model = InferenceSession(model_path, sess_options=sess_options)

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=model, model_path=model_path)
    ]
)
def depth_pro(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ]
) -> tuple[
    Annotated[
        ndarray,
        Parameter.DepthMap(description="Metric depth map with shape (H, W).")
    ],
    Annotated[
        float,
        Parameter.Generic(description="Focal length in pixels.")
    ]
]:
    """
    Estimate metric depth from an image using Apple Depth Pro.
    """
    # Preprocess image
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    normalized_tensor = F.normalize(
        image_tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    image_batch = normalized_tensor[None]
    # Run ONNX model inference
    outputs = model.run(None, { "pixel_values": image_batch.numpy() })
    depth_tensor = outputs[0].squeeze()
    focal_length = float(outputs[1].squeeze())
    # Return
    return depth_tensor, focal_length

def _visualize_depth(depth: ndarray) -> Image.Image:
    """
    Colorize a depth array using OpenCV's COLORMAP_INFERNO heatmap.
    """
    depth_range = depth.max() - depth.min()
    depth_normalized = (depth - depth.min()) / depth_range
    depth_uint8 = (depth_normalized * 255).astype(uint8)
    depth_colored = applyColorMap(depth_uint8, COLORMAP_INFERNO)
    depth_colored = cvtColor(depth_colored, COLOR_BGR2RGB)
    return Image.fromarray(depth_colored)

if __name__ == "__main__":
    from pathlib import Path
    # Predict
    image_path = Path(__file__).parent / "demo" / "room.jpg"
    image = Image.open(image_path)
    depth, focal_length = depth_pro(image)
    print(f"Focal length: {focal_length:.1f} pixels. Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    # Visualize
    depth_img = _visualize_depth(depth)
    depth_img.show()