#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def for_(number: float) -> float:
    """
    Test support for for-loops.
    """
    for i in range(10):
        number += i
    return number