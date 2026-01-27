#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# NOTE: In order to run and/or compile this model, you must clone Depth Anything into the current working directory.
# Run `git clone https://github.com/LiheYoung/Depth-Anything.git` then run/compile this script.

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "opencv-python-headless", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from numpy import ndarray, uint8
from pathlib import Path
from PIL import Image
import sys
from torch import inference_mode, randn
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from typing import Annotated

# Import Depth Anything
sys.path.insert(0, str(Path.cwd() / "Depth-Anything"))
from depth_anything.dpt import DepthAnything

# Create model
model = DepthAnything.from_pretrained(
    "LiheYoung/depth_anything_vitl14",
    config={ "localhub": False }
).eval()
INPUT_SIZE = 518

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "opencv-python-headless")
        .run_commands("git clone https://github.com/LiheYoung/Depth-Anything.git"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, INPUT_SIZE, INPUT_SIZE)],
        )
    ]
)
@inference_mode()
def depth_anything_large(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ]
) -> Annotated[
    ndarray,
    Parameter.DepthMap(description="Metric depth tensor with shape (H,W).")
]:
    """
    Estimate metric depth from an image using Depth Anything (large).
    """
    # Resize image
    width, height = image.size
    ratio = min(INPUT_SIZE / width, INPUT_SIZE / height)
    scaled_width = int(width * ratio)
    scaled_height = int(height * ratio)
    image = F.resize(image, [scaled_height, scaled_width])
    # Pad image to square
    padding = (0, 0, INPUT_SIZE - scaled_width, INPUT_SIZE - scaled_height)
    image = image.convert("RGB")
    image = F.pad(image, padding, fill=0)
    # Convert to tensor and normalize
    image_tensor = F.to_tensor(image)
    normalized_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run model
    depth_tensor = model(normalized_tensor[None])
    # Upsample depth map
    depth_resized = interpolate(
        depth_tensor[None,:,:scaled_height,:scaled_width],  # remove padding
        size=(height, width),
        mode="bilinear"
    )
    depth = depth_resized[0,0].cpu().numpy() # (H,W)
    # Return
    return depth

def _visualize_depth(depth: ndarray) -> Image.Image:
    """
    Colorize a depth array using OpenCV's COLORMAP_INFERNO heatmap.
    """
    from cv2 import applyColorMap, cvtColor, COLOR_BGR2RGB, COLORMAP_INFERNO
    depth_range = depth.max() - depth.min()
    depth_normalized = (depth - depth.min()) / depth_range
    depth_uint8 = (depth_normalized * 255).astype(uint8)
    depth_colored = applyColorMap(depth_uint8, COLORMAP_INFERNO)
    depth_colored = cvtColor(depth_colored, COLOR_BGR2RGB)
    return Image.fromarray(depth_colored)

if __name__ == "__main__":
    # Predict
    image_path = Path(__file__).parent / "demo" / "city.jpg"
    image = Image.open(image_path)
    depth = depth_anything_large(image)
    # Visualize
    depth_img = _visualize_depth(depth)
    depth_img.show()
