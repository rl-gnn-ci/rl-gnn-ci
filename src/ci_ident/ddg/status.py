"""Per-node status codes of the identification state.

The status of every node is part of the observation (one embedding row per
code), so the numbering is part of the model interface and must not change.
"""

from __future__ import annotations

from enum import Enum


class NodeStatus(Enum):
    NOT_SELECTED = 0                 # free, CI-admissible
    NOT_SELECTED_INPUT = 1           # free, currently an input of the candidate
    SELECTED = 2                     # in the candidate
    SELECTED_OUTPUT = 3              # in the candidate, produces an output
    SELECTED_PERM_OUTPUT = 4         # in the candidate, output that can never be absorbed
    ALREADY_SELECTED = 5             # in a committed CI
    ALREADY_SELECTED_PERM_INPUT = 6  # in a committed CI, input of the candidate
    INPUT_NODE = 7                   # source of the DDG (block input)
    INPUT_NODE_PERM_INPUT = 8        # source of the DDG, input of the candidate
    INVALID = 9                      # inadmissible operation (memory, control, ...)
    INVALID_PERM_INPUT = 10          # inadmissible, input of the candidate


NOT_SELECTED_GROUP = frozenset({NodeStatus.NOT_SELECTED, NodeStatus.NOT_SELECTED_INPUT})
SELECTED_GROUP = frozenset({NodeStatus.SELECTED, NodeStatus.SELECTED_OUTPUT,
                            NodeStatus.SELECTED_PERM_OUTPUT})
ALREADY_SELECTED_GROUP = frozenset({NodeStatus.ALREADY_SELECTED,
                                    NodeStatus.ALREADY_SELECTED_PERM_INPUT})
INPUT_NODE_GROUP = frozenset({NodeStatus.INPUT_NODE, NodeStatus.INPUT_NODE_PERM_INPUT})
INVALID_GROUP = frozenset({NodeStatus.INVALID, NodeStatus.INVALID_PERM_INPUT})

#: Nodes that can never join the candidate under construction.
FIXED_GROUP = ALREADY_SELECTED_GROUP | INVALID_GROUP | INPUT_NODE_GROUP

NUM_STATUSES = len(NodeStatus)
