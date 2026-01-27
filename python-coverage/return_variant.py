#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

# The Muna compiler imposes restrictions on the entrypoint function (i.e. function decorated with `@compile`).
# One such restriction is that the function **must not** return a union/variant type.
# But this restriction is a bit fake. The compiler does in fact support variant types.
# We just don't like it because it messes with signature inference.

# TL;DR: You can't annotate the entrypoint to return a union, but you can return a union 😇

@compile()
def return_variant(as_num: bool) -> str:
    """
    Test support for returning a variant.
    """
    if as_num:
        result = 1
    else:
        result = "one"
    return result