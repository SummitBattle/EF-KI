from MCTS_MS import *


def start_MCTS(game_state, done=False, parent_node=None, action_index=None, depth=3, reward=0):
    """Runs MCTS with Minimax and returns the next tree and action."""

    node = Node(game_state, done, parent_node, action_index,reward)

    node.explore(depth)



    next_tree, next_action = node.next()

    if next_tree is None:
        raise ValueError("MCTS failed to find the next move.")

    return next_action


