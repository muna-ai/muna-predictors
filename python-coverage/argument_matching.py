#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Python supports invoking functions with positional and keyword 
# arguments, and the latter can be provided out-of-order. 
# Our compiler supports all these forms of arguments.

from muna import compile, Parameter
from typing import Annotated

@compile()
def argument_matching(
    a: Annotated[float, Parameter.Numeric(description="First number.")],
    b: Annotated[float, Parameter.Numeric(description="Second number with default.")] = 10.0,
    c: Annotated[float, Parameter.Numeric(description="Third number with default.")] = 42.0
) -> Annotated[float, Parameter.Numeric(description="Sum of a, b, and c.")]:
    """
    Test support for matching regular, positional-only, and keyword arguments.
    """
    # Positional arguments
    sub_all_positional = _sub(12.0, 9.0)
    # Default argument
    sub_one_default = _sub(12.0)
    # Keyword arguments
    sub_with_keyword = _sub(a=20.0, b=5.0)
    # Keyword arguments out of order
    sub_keyword_reversed = _sub(b=3.0, a=10.0)
    # Positional and keyword arguments
    sub_mixed = _sub(15.0, b=8.0)
    # Keyword and default arguments
    sub_keyword_optional = _sub(a=25.0)
    # Test with variables instead of constants
    sub_with_var = _sub(a, b)
    sub_with_var_default = _sub(a)
    # Return
    return (
        a +
        b +
        c +
        sub_all_positional +
        sub_one_default +
        sub_with_keyword +
        sub_keyword_reversed +
        sub_mixed +
        sub_keyword_optional +
        sub_with_var +
        sub_with_var_default
    )

def _sub(a, b=10.):
    return a - b

if __name__ == "__main__":
    print(argument_matching(3))