#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile

@compile()
def argument_default_value_hetero(items: list[str]="nothing here") -> int:
    """
    Test support for arguments with default values of a different type.
    """
    # `items` should be std::variant<std::string, std::vector<std::string>>
    return len(items)