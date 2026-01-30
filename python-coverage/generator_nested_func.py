#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Generator that uses a nested helper function.
# The nested function is NOT a generator - it uses regular return.
# This tests that returns inside the nested function emit `return`, not `co_return`.

from muna import compile
from typing import Iterator

@compile()
def generator_nested_func(
    values: list[int],
    multiplier: int
) -> Iterator[int]:
    """
    Test generator with nested non-generator function.
    """
    # Define a nested function that is not a generator
    def transform(x: int) -> int:
        # This is a regular function, not a generator
        # Should emit `return`, not `co_return`
        if x < 0:
            return 0
        return x * multiplier
    # Yield transformed values
    for val in values:
        yield transform(val)