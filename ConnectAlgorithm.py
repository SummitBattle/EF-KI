from MCTS_MS import Node
import logging
from utility_functions import HUMAN_PLAYER, AI_PLAYER


def start_MCTS(game_state, done=False, parent_node=None, action_index=None, depth=3, reward=0, playerMove=None,
               human_starts=True):
    """
    Runs MCTS with Minimax and returns the next action.

    Assumes that:
      - `game_state` is a BitBoard instance.
      - The BitBoard instance contains the current player in game_state.current_player.
      - The Node class is implemented to work with the bitboard state.

    Parameters:
      - game_state: current state (BitBoard instance)
      - done: whether the game is over (default False)
      - parent_node: a previously built search tree (if any)
      - action_index: index of the action that led to this state (if any)
      - depth: exploration depth for the MCTS
      - reward: (unused here) initial reward value for the node
      - playerMove: the last move made by the player (if applicable)
      - human_starts: boolean indicating if human started the game (default True)

    Returns:
      - next_action: the column (0-indexed) that MCTS recommends
      - next_tree: the subtree corresponding to that move (for tree reuse)
    """
    # Use the game state's current player rather than deciding based on human_starts.
    player = game_state.current_player

    if done:
        logging.warning("The game is already over. No further moves can be made.")
        return None, None

    # Initialize the root node. If a parent node is provided, reuse it.
    root = parent_node if parent_node else Node(game_state, done, None, action_index, player)

    # If we already have a parent node and the human (or opponent) has made a move,
    # update the root node accordingly.
    if parent_node and playerMove is not None:
        root = root.movePlayer(playerMove)

    # Perform exploration using the provided depth.
    root.explore(depth)

    # Choose the best next move and get the corresponding subtree.
    next_tree, next_action = root.next()

    if next_tree is None or next_action is None:
        logging.error("MCTS failed to find the next move.")
        return None, None  # Handle failure gracefully

    logging.info(f'Next Action: {next_action}')
    return next_action, next_tree
