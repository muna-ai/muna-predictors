#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "torchvision"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from torch import argmax, inference_mode, softmax, randn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import functional as F
from typing import Annotated

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).eval()

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, 224, 224)]
        ),
    ]
)
@inference_mode()
def mobilenet_v2(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")]
) -> tuple[
    Annotated[str, Parameter.Generic(description="Classification label.")],
    Annotated[float, Parameter.Generic(description="Classification score.")]
]:
    """
    Classify an image with MobileNet v2.
    """
    # Preprocess
    image = image.convert("RGB")
    image = F.resize(image, 224)
    image = F.center_crop(image, 224)
    image_tensor = F.to_tensor(image)
    normalized_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run model
    logits = model(normalized_tensor[None])
    # Post-process outputs
    scores = softmax(logits, dim=1)
    idx = argmax(scores, dim=1)
    score = scores[0,idx].item()
    label = weights.meta["categories"][idx]
    # Return
    return label, score

if __name__ == "__main__":
    from pathlib import Path
    # Predict
    image_path = Path(__file__).parent / "demo" / "cat.jpg"
    image = Image.open(image_path)
    label, score = mobilenet_v2(image)
    # Print
    print(label, score)