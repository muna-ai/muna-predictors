#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from typing import Iterator

@compile()
def generator(sentence: str) -> Iterator[str]:
    """
    Test compiling generator functions.
    """
    parts = sentence.split()
    for part in parts:
        yield part