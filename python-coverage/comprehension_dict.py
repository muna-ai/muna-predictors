#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def dict_comprehension(count: int) -> dict:
    """
    Test support for dictionary comprehensions.
    """
    return { x: f"The number is {x}" for x in range(count) }