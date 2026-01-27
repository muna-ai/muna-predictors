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
from torch import randn, argmax, softmax
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.transforms import functional as F
from typing import Annotated

weights = SqueezeNet1_1_Weights.DEFAULT
model = squeezenet1_1(weights=weights).eval()

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, 224, 224)]
        )
    ]
)
def squeezenet(
    image: Annotated[Image.Image, Parameter.Generic(description="Input image.")]
) -> tuple[
    Annotated[str, Parameter.Generic(description="Classification label.")],
    Annotated[float, Parameter.Generic(description="Classification score.")]
]:
    """
    Classify an image with SqueezeNet 1.1.
    """
    # Preprocess image
    image = F.resize(image, 256)
    image = image.convert("RGB")
    image = F.center_crop(image, 224)
    image_tensor = F.to_tensor(image)
    normalized_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run model
    logits = model(normalized_tensor[None])
    scores = softmax(logits, dim=1)
    idx = argmax(scores, dim=1)
    score = scores[0, idx].item()
    label = weights.meta["categories"][idx]
    # Return
    return label, score

if __name__ == "__main__":
    from pathlib import Path
    # Predict
    image_path = Path(__file__).parent / "demo" / "cat.jpg"
    image = Image.open(image_path)
    label, score = squeezenet(image)
    # Print
    print(label, score)