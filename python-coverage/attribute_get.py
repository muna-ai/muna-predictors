# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from PIL import Image

@compile()
def attribute_get(image: Image.Image) -> tuple[int, int]:
    """
    Test support for `getattr`.
    """
    return image.size