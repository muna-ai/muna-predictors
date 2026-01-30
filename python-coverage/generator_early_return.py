#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from typing import Iterator

@compile()
def generator_early_return(count: int) -> Iterator[int]:
    """
    Test generator with early return before yield.
    """
    # Early return if count is non-positive
    if count <= 0:
        return # this should be emitted as `co_return;` in C++
    # Yield numbers
    for i in range(count):
        yield i