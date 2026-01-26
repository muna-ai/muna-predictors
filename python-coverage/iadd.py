#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
import numpy as np

@compile()
def iadd() -> int:
    """
    Test emitting the correct C++ call for in-place add.
    """
    current_length = 1
    for _ in range(10):
        current_length +=  1
    return current_length