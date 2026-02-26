#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# This script classifies images using the MobileNet v2 model.
# It can then be compiled into a self-contained library for 
# deployment, accelerated by Qualcomm NPUs. First, run the 
# script:
# ```
# $ uv run mobilenet_v2_qnn.py
# ```
# 
# Next, transpile it into a C++ library for deployment:
# ```
# $ pip install cmake muna
# $ muna transpile mobilenet_v2_qnn.py --output cpp
# ```
#
# Finally, compile and run the C++ library and demo:
# ```
# $ cd cpp && cmake -B build && cmake --build build --parallel
# $ ./mobilenet_v2_qnn --image demo/cat.jpg
# ```
#
# With this, you can keep a simple Python-first workflow 
# while targeting a wide range of Qualcomm processors for AI inference.

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "torchvision"]
# ///

from muna import compile, Sandbox
from muna.beta import QnnInferenceMetadata
from PIL import Image
from torch import randn, argmax, softmax
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms.functional import center_crop, normalize, resize, to_tensor

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).eval()

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu"),
    metadata=[
        QnnInferenceMetadata(
            model=model,
            model_args=[randn(1, 3, 224, 224)],
            backend="gpu"
        )
    ]
)
def mobilenet_v2_qnn(image: Image.Image) -> tuple[str, float]:
    """
    Classify an image with MobileNet v2.
    """
    # Preprocess image
    image = resize(image, 256)
    image = image.convert("RGB")
    image = center_crop(image, 224)
    image_tensor = to_tensor(image)
    normalized_tensor = normalize(
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
    label, score = mobilenet_v2_qnn(image)
    # Print
    print(label, score)