#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def fstring(name: str) -> str:
    """
    Test string interpolation with f-strings.
    """
    return f"Hey there {name}! We're glad you're trying out Function and we hope you like it 😉"