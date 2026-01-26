#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# Once lowered, the types of `result_a` and `result_b` MUST match.
# Specifically, we need to see a deterministic ordering of variant member types.

@compile()
def list_concat_hetero(name: str, age: int) -> list:
    """
    Test heterogenous list concatenation.
    """
    names = [name, name, name, name]
    ages = [age, age, age, age]
    result_a = names + ages
    result_b = ages + names # C++ types for `result_a` and `result_b` MUST match
    return len(result_a) + len(result_b)