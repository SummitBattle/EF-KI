import logging
from MCTS_MS import Node
from utility_functions import blockOrWinMove
# Parameters for the Q-network
INPUT_SIZE = 42  # Flattened 6x7 Connect 4 board
OUTPUT_SIZE = 7  # Number of columns for possible moves
Q_NETWORK_PATH = "q_network.pth"  # Path to the pretrained Q-network file
from board import BitBoard


def start_MCTS(game_state, done=False, parent_node=None, action_index=None, playerMove=None, human_starts=True):
    """
    Runs MCTS with Q-learning guidance and returns the next action.

    Assumes that:
      - `game_state` is a BitBoard instance.
      - The BitBoard instance contains the current player in game_state.current_player.
      - The Node class is implemented to work with the bitboard state and Q-network.

    Parameters:
      - game_state: current state (BitBoard instance)
      - done: whether the game is over (default False)
      - parent_node: a previously built search tree (if any)
      - action_index: index of the action that led to this state (if any)
      - playerMove: the last move made by the player (if applicable)
      - human_starts: boolean indicating if human started the game (default True)
      - min_rollouts: minimum number of rollouts for exploration (default 100)
      - epsilon: probability for random move selection in epsilon-greedy exploration (default 0.1)

    Returns:
      - next_action: the column (0-indexed) that MCTS recommends
      - next_tree: the subtree corresponding to that move (for tree reuse)
    """

    # Initialize the root node. Reuse a parent node if provided.
    root = parent_node if parent_node else Node(game_state, done, None, action_index, human_starts)

    # If a parent node exists and a move was made by the player, update the tree accordingly.
    if parent_node and playerMove is not None:
        root = root.movePlayer(playerMove)
    print("ROOT GAME STATE")
    root.game_state.print_board()
    root.game_state.print_board()
    root.game_state.print_board()
    root.game_state.print_board()
    # Perform exploration using MCTS with Q-learning guidance.
    root.explore()


    # Choose the best next move and get the corresponding subtree.
    try:
        next_tree, next_action = root.next()
    except ValueError as e:
        logging.error(f"MCTS failed to find the next move: {e}")
        return None, None


    logging.info(f"Next Action: {next_action}")
    return next_action, next_tree