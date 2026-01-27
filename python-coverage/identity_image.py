#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from PIL import Image

@compile()
def identity_image(image: Image.Image) -> Image.Image:
    """
    Test returning an input image argument as-is.
    """
    return image