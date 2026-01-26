#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Muna has enhanced support for enumeration arguments. During codegen, the compiler 
# will extract the enumeration members into the predictor signature.
# On `https://muna.ai`, this is how we show dropdowns for enumeration parameters.
# NOTE: The enumeration member values **MUST** be all be strings or integers (no mixing either!).

from enum import IntEnum
from muna import compile

class Direction (IntEnum):
    North = 0
    East = 1
    South = 2
    West = 3

@compile()
def argument_enum(direction: Direction) -> float:
    """
    Test enumeration argument support.
    """
    return direction * 90.