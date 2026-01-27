#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def while_(number: float) -> float:
    """
    Test while-loop support.
    """
    while number > 2:
        number = number - 1
    return number