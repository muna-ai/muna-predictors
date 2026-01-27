#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def quoted_string() -> str:
    """
    Testing lowering strings with quotes.
    """
    return 'He said "What?"'