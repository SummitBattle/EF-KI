from MCTS_MS import *  # Make sure Node and related classes are imported


def start_MCTS(game_state, done=False, parent_node=None, action_index=None, depth=3, reward=0):
    """Runs MCTS with Minimax and returns the next action."""

    # Initialize the root node of the search tree
    if parent_node is None:
        node = Node(game_state, done, None, action_index, reward)
    else:
        node = parent_node

    # If the game is already over, return None or some other fallback behavior
    if done:
        raise ValueError("The game is already over. No further moves can be made.")

    # Perform the exploration (including simulations, UCT updates, and backpropagation)
    node.explore(depth)

    # Get the next tree and action to take based on the visits to child nodes
    next_tree, next_action = node.next()

    if next_tree is None:
        raise ValueError("MCTS failed to find the next move.")

    logging.info(f'Next Action: {next_action}')
    return next_action, next_tree
