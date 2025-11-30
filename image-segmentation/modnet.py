#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "torchvision"]
# ///

from huggingface_hub import hf_hub_download
from muna import compile, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from onnxruntime import InferenceSession
from PIL import Image
from torchvision.transforms import functional as F

# Instantiate model
model_path = hf_hub_download(
    repo_id="Xenova/modnet",
    filename="onnx/model.onnx"
)
model = InferenceSession(model_path)

@compile(
    tag="@cuhk/modnet",
    description="Realtime trimap-free portrait matting.",
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("onnxruntime", "huggingface_hub"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=model, model_path=model_path)
    ]
)
def predict(image: Image.Image) -> Image.Image:
    # Compute scale factor
    dst_height = 512
    dst_width = int(image.width / image.height * dst_height)
    dst_width = dst_width // 16 * 16
    # Preprocess image
    rgb_image = image.convert("RGB")
    resized_image = F.resize(rgb_image, (dst_height, dst_width))
    image_tensor = F.to_tensor(resized_image)
    image_batch = image_tensor[None]
    # Run model
    mask = model.run(None, { "input": image_batch.numpy() })[0]
    # Get output
    mask_image = F.to_pil_image(mask.squeeze())
    mask_image = F.resize(mask_image, (image.height, image.width))
    # Build result
    result = image.convert("RGBA")
    result.putalpha(mask_image)
    # Return
    return result

if __name__ == "__main__":
    image = Image.open("test/media/headshot.jpg")
    result = predict(image)
    result.show()