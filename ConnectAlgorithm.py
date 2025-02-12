from MCTS_MS import *  # Ensure Node and related classes are imported
import logging

def start_MCTS(game_state, done=False, parent_node=None, action_index=None, depth=3, reward=0):
    """Runs MCTS with Minimax and returns the next action."""

    # If the game is over, return None or an appropriate fallback action
    if done:
        logging.warning("The game is already over. No further moves can be made.")
        return None, None

    # Initialize the root node of the search tree
    node = parent_node if parent_node is not None else Node(game_state, done, None, action_index)

    # Perform exploration (including simulations, UCT updates, and backpropagation)
    node.explore(depth)

    # Get the best next move and corresponding subtree
    next_tree, next_action = node.next()

    if next_tree is None or next_action is None:
        logging.error("MCTS failed to find the next move.")
        return None, None  # Handle failure gracefully

    logging.info(f'Next Action: {next_action}')
    return next_action, next_tree
