# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def assignment(num: int) -> int:
    """
    Test variable assignment.
    """
    x = num
    return x