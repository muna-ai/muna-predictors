#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from math import pi

@compile()
def return_global() -> float:
    """
    Test returning a global variable.
    """
    return pi