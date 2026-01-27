#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def return_variant_of_tuples(friendly: bool) -> str: # should be std::variant<bool, std::tuple<std::string, int32_t>> in C++
    """
    Return a variant of tuples.
    """
    if friendly:
        return "Hello!", 25
    else:
        return False