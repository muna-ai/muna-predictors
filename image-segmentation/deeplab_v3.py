#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["muna", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from numpy import int64
from numpy.typing import NDArray
from PIL import Image
from torch import from_numpy, inference_mode, randn, Tensor, randint
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks
from typing import Annotated

# Create the model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()
INPUT_SIZE = 520

@compile(
    tag="@google/deeplab-v3",
    description="Segment an image with DeepLab v3.",
    sandbox=Sandbox().pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, INPUT_SIZE, INPUT_SIZE)],
            output_keys=["out", "aux"]
        )
    ]
)
@inference_mode()
def segment_image(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")]
) -> Annotated[NDArray[int64], Parameter.Generic(description="Segmentation mask tensor with shape (H,W).")]:
    """
    Segment an image with DeepLab v3.
    """
    # Preprocess image
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run model
    output_dict: dict[str, Tensor] = model(image_tensor[None])
    # Post-process outputs
    logits = output_dict["out"][0]
    mask = logits.argmax(dim=0)
    # Return
    return mask.numpy()

def _visualize_segmentation_mask(
    image: Image.Image,
    mask: NDArray[int64],
    alpha: float=0.6
) -> Image.Image:
    """
    Visualize a segmentation mask.
    """
    # Check unique classes
    mask_tensor = from_numpy(mask)
    class_ids = mask_tensor.unique(sorted=True)
    class_ids = class_ids[class_ids != 0]
    num_classes = class_ids.numel()
    if num_classes == 0:
        return image
    # Visualize
    class_masks = mask_tensor[None] == class_ids[:, None, None] # (M,H,W)
    image_tensor = F.to_tensor(image)
    # Generate random colors for each class
    random_colors = [tuple(randint(0, 256, (3,)).tolist()) for _ in range(num_classes)]
    result_tensor = draw_segmentation_masks(
        image_tensor,
        class_masks,
        alpha=alpha,
        colors=random_colors
    )
    result = F.to_pil_image(result_tensor)
    # Return
    return result

if __name__ == "__main__":
    from pathlib import Path
    # Predict
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    mask = segment_image(image)
    # Visualize
    annotated_image = _visualize_segmentation_mask(image, mask)
    annotated_image.show()