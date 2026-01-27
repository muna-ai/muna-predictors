#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def return_constant() -> str:
    """
    Test returning an inline constant.
    """
    return "Hello from Function"