import random
import time
import math
from copy import deepcopy
import concurrent.futures

from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue


def copy_board(bitboard):
    """Fast board copy: use a built-in copy method if available."""
    if hasattr(bitboard, 'copy'):
        return bitboard.copy()
    return deepcopy(bitboard)


def count_moves(bitboard):
    """
    Count the number of moves for each player using the bitboard representation.
    We assume:
      - bitboard.board1 holds the human moves ('x')
      - bitboard.board2 holds the AI moves ('o')
    """
    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")
    return ai_moves, human_moves


def center_bias(game_state, action):
    """
    Compute a bonus for moves closer to the center column of a Connect-4 board.
    Uses exponential scaling for higher center bias.
    """
    if not hasattr(game_state, 'width'):
        return 0.0  # No bias if board dimensions are undefined

    try:
        col = action  # Only the column is relevant in Connect-4
    except Exception:
        return 0.0

    center_col = game_state.width // 2
    dist = abs(col - center_col)
    max_distance = center_col  # Maximum distance is from the farthest column to the center
    bonus_factor = 20000.0  # Adjust this to control bias strength

    # Exponential decay for stronger bias near the center
    bonus = bonus_factor * (1 - (dist / max_distance) ** 2)
    return bonus


def rollout_simulation(game_state, minimax_depth):
    """
    A standalone rollout function that performs a simulation from a given board state.
    This function is designed to be run in parallel.
    """
    board_state = copy_board(game_state)

    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    # --- First Move with Minimax ---
    best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    # Switch current player for the random rollout phase.
    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

    # --- Random Rollout Phase ---
    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5  # Draw value if no moves available.
        action = random.choice(valid_moves)
        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)


class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'reward', 'starting_player')

    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # action -> Node
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # BitBoard instance
        self.done = done
        self.action_index = action_index  # The move that led to this node (assumed to be (row, col))
        self.c = 1.41  # Exploration constant (or math.sqrt(2))
        self.reward = 0.0
        self.starting_player = starting_player

    def getUCTscore(self):
        """
        Calculate the UCT score with center bias applied.
        """
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        bias = 0.0
        if self.action_index is not None:
            bias = center_bias(self.game_state, self.action_index)

        # Weight the bias to balance it with node values and exploration
        bias_weight = 0.5  # Adjust to control the effect of center bias
        return (self.node_value / self.visits) + (bias_weight * bias) + self.c * math.sqrt(
            math.log(parent_visits) / self.visits)

    def create_child_nodes(self):
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            # Play the move; assume play_move toggles the current_player automatically.
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=3, min_rollouts=50000000, min_time=0.0, max_time=8.0, batch_size=8):
        """
        Explore the tree using parallel rollouts.
        Instead of running each rollout synchronously, we schedule batches of rollouts in parallel.
        """
        start_time = time.perf_counter()
        rollouts = 0

        rand_choice = random.choice
        get_time = time.perf_counter
        batch = []  # list of tuples: (future, node)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                elapsed = get_time() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    break

                current = self
                # --- Selection Phase ---
                while current.child_nodes:
                    best_score = -float('inf')
                    best_children = []
                    for child in current.child_nodes.values():
                        score = child.getUCTscore()
                        if score > best_score:
                            best_score = score
                            best_children = [child]
                        elif score == best_score:
                            best_children.append(child)
                    current = rand_choice(best_children)

                # --- Expansion Phase ---
                if not current.done:
                    current.create_child_nodes()
                    if current.child_nodes:
                        current = rand_choice(list(current.child_nodes.values()))

                # Schedule the rollout simulation in parallel.
                future = executor.submit(rollout_simulation, current.game_state, minimax_depth)
                batch.append((future, current))
                rollouts += 1

                # Process batch once full.
                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()
                        except Exception:
                            reward = 0.0  # Fallback if simulation fails.
                        # --- Backpropagation Phase ---
                        node_to_update = node
                        while node_to_update:
                            node_to_update.visits += 1
                            node_to_update.node_value += reward
                            node_to_update = node_to_update.parent
                    batch.clear()

            # Process any remaining futures.
            for future, node in batch:
                try:
                    reward = future.result()
                except Exception:
                    reward = 0.0
                node_to_update = node
                while node_to_update:
                    node_to_update.visits += 1
                    node_to_update.node_value += reward
                    node_to_update = node_to_update.parent

        print(f"\nTotal rollouts performed: {rollouts}")
        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """
        Rollout with center bias applied during move selection.
        """
        board_state = copy_board(self.game_state)

        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

        best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
        if best_move is not None:
            board_state.play_move(best_move)
            if gameIsOver(board_state):
                return EndValue(board_state, AI_PLAYER)

        current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

        # Rollout with center bias applied
        while not gameIsOver(board_state):
            valid_moves = board_state.get_valid_moves()
            if not valid_moves:
                return 0.5  # Draw value if no moves are available

            # Calculate biases for all valid moves
            move_biases = [center_bias(board_state, move) for move in valid_moves]
            # Choose a move with weighted random selection
            action = random.choices(valid_moves, weights=move_biases, k=1)[0]

            board_state.current_player = current_player
            board_state.play_move(action)
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(board_state, AI_PLAYER)

    def next(self):
        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children available. Ensure exploration has been performed.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)
        print(f"\nSelected move: {best_child.action_index}")
        print("Board state for the selected move:")
        best_child.game_state.print_board()
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        if playerMove in self.child_nodes:
            new_root = self.child_nodes[playerMove]
        else:
            new_board = copy_board(self.game_state)
            new_board.play_move(playerMove)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root
        return new_root
