#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def func_reuse() -> int:
    """
    Test that we reuse previously-lowered functions when generating code.
    """
    # This should lower `some_supporting_func`
    first_num = _some_supporting_func(3)
    # This should not lower `some_supporting_func`
    second_num = _some_wrapper_of_some_supporting_func(5)
    # Return
    return first_num + second_num

def _some_wrapper_of_some_supporting_func(x: int) -> int:
    return _some_supporting_func(x - 2)

def _some_supporting_func(x: int) -> int:
    return x * 2