#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Muna has enhanced support for literal arguments. During codegen, the compiler 
# will extract the literal members into the predictor signature.
# On `https://muna.ai`, this is how we show dropdowns for literal parameters.
# NOTE: The literal member values **MUST** be all be strings or integers.

from muna import compile
from typing import Literal

Direction = Literal["north", "east", "south", "west"]

@compile()
def argument_literal(direction: Direction) -> float:
    """
    Test literal argument support.
    """
    match direction:
        case "north":   return 0
        case "east":    return 90
        case "south":   return 180
        case "west":    return 270