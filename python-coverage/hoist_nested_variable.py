#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# One major difference between Python and C++ is how each language handles scopes.
#
# In Python, there is no concept of scopes within a function. If you define a variable 
# within an indented block (e.g. `if` or `while` statement body), the variable remains 
# accessible outside that scope.
#
# But in C++, variables defined within a scope cannot be accessed outside that scope.
# So to emulate Python's behaviour in C++, we simply hoist all variables that are 
# accessed outside their defining scope. We then promote the type of the hoisted 
# variable accordingly (e.g. `T` becomes `std::optional<T>`).

from muna import compile
import numpy as np

@compile()
def hoist_nested_variable() -> np.int64:
    """
    Hoist a nested variable.
    """
    for i in range(2):
        result = 20
    return result