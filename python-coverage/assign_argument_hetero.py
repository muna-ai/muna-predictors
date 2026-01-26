# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def assign_argument_hetero(x: int) -> str:
    """
    Assigning heterogenous values to a function argument variable.
    """
    # Since `x` is assigned to a string, its C++ type needs to be 
    # updated to `std::variant<int32_t, std::string>`. This update 
    # must be reflected in the function prototype.
    x = "hello world"
    return x