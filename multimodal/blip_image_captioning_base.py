#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from PIL import Image
from torch import inference_mode, int64, randint, randn
from torch.export import Dim
from torchvision.transforms.functional import normalize, resize, to_tensor
from transformers import AutoTokenizer, BlipForConditionalGeneration

# Load model & tokenizer
REPO_ID = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(REPO_ID).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("onnxruntime", "transformers"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[
                randn(1, 3, 384, 384),
                randint(0, 30524, (1, 10), dtype=int64),
            ],
            input_shapes=[
                (1, 3, 384, 384),
                (1, Dim("seq_len", min=1, max=128)),
            ]
        ),
    ]
)
@inference_mode()
def blip_image_captioning_base(image: Image.Image) -> str:
    """
    Generate a caption for an image using Salesforce BLIP (base).
    """
    image = image.convert("RGB")
    image = resize(image, [384, 384])
    image_tensor = to_tensor(image)
    normalized = normalize(
        image_tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    pixel_values = normalized[None]
    output_ids = model.generate(pixel_values, max_new_tokens=20)
    caption_ids = output_ids[0]
    return tokenizer.decode(caption_ids, skip_special_tokens=True)

if __name__ == "__main__":
    image = Image.open("test/media/cat.jpg")
    result = blip_image_captioning_base(image)
    print(result)