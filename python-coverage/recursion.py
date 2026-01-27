#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# Muna has limited support for recursive functions, primarily due to challenges in type propagation.
# Recursive functions **must** have return type annotations, and for now those types must be simple.
# In the future, we might add proper support for recursion with fixed point iteration or Algorithm W.

@compile()
def factorial(n: int) -> int:
    """
    Test support for recursion.
    """
    match n:
        case 0: return 1
        case 1: return 1
        case _: return n * factorial(n - 1)