#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def return_none() -> None:
    """
    Test returning nothing.
    """
    pass