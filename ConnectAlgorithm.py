from MCTS_MS import Node
import logging
from board import printBoard
from board import HUMAN_PLAYER
from board import AI_PLAYER


def start_MCTS(game_state, done=False, parent_node=None, action_index=None, depth=3, reward=0, playerMove=None, human_starts=HUMAN_PLAYER):
    """Runs MCTS with Minimax and returns the next action."""
    # If the game is over, return None or an appropriate fallback action

    if human_starts:
        player = HUMAN_PLAYER
    else:
        player = AI_PLAYER

    if done:
        logging.warning("The game is already over. No further moves can be made.")
        return None, None

    # Initialize the root node of the search tree
    root = parent_node if parent_node else Node(game_state, done, None, action_index, player)

    # Update the root with the player's move if necessary
    if parent_node and playerMove is not None:
        root = root.movePlayer(playerMove)


    # Perform exploration (including simulations, UCT updates, and backpropagation)
    root.explore(depth)

    # Get the best next move and corresponding subtree
    next_tree, next_action = root.next()

    if next_tree is None or next_action is None:
        logging.error("MCTS failed to find the next move.")
        return None, None  # Handle failure gracefully

    logging.info(f'Next Action: {next_action}')
    return next_action, next_tree