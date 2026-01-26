# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def assign_local_variant() -> str:
    """
    Assigning heterogenous values to a local variable.
    """
    x = 10
    x = "hello world"
    return x