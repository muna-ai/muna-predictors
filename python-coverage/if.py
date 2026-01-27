#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def if_(score: float) -> str:
    """
    Test if-statement support.
    """
    if score < 0.2:
        grade = "low"
    elif score < 0.8:
        grade = "medium"
    else:
        grade = "high"
    return grade