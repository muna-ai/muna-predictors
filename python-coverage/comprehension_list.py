#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def list_comprehension(count: int) -> list:
    """
    Test support for list comprehensions.
    """
    return [f"The number is {x}" for x in range(count)]