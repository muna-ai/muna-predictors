#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def set_comprehension() -> int:
    """
    Test support for set comprehensions.
    """
    names = ["Yusuf", "Terri", "Rhea", "Muna", "Terri"]
    unique_names = { "hello " + name for name in names }
    return len(unique_names)