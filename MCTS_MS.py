import random
import time
import math
from copy import deepcopy
import concurrent.futures

from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue


def copy_board(bitboard):
    """Use a fast board copy method if available."""
    return bitboard.copy() if hasattr(bitboard, 'copy') else deepcopy(bitboard)


def count_moves(bitboard):
    """
    Count the moves for each player.
    Assumes:
      - bitboard.board1 holds the human moves ('x')
      - bitboard.board2 holds the AI moves ('o')
    """
    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")
    return ai_moves, human_moves


def center_bias(game_state, action):
    """
    Compute a bonus for moves closer to the center.
    Assumes `action` is an integer representing the column.
    """
    if not hasattr(game_state, 'width'):
        return 0.0

    col = action  # Action is simply the column.
    center_col = game_state.width // 2
    dist = abs(col - center_col)
    max_distance = center_col
    bonus_factor = 5.0
    # Exponential decay: moves nearer to center get a higher bonus.
    return bonus_factor * (1 - (dist / max_distance) ** 2)


def rollout_simulation(game_state, minimax_depth):
    """
    Perform a simulation from the given board state.
    This function is intended to run in parallel.
    """
    board_state = copy_board(game_state)

    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    # --- First Move using Minimax ---
    best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    # --- Random Rollout Phase ---
    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER
    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5  # Draw if no moves.
        action = random.choice(valid_moves)
        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)


class Node:
    __slots__ = (
        'parent',
        'child_nodes',
        'visits',
        'node_value',
        'game_state',
        'done',
        'action_index',
        'c',
        'reward',
        'starting_player'
    )

    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # action -> Node
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # BitBoard instance
        self.done = done
        self.action_index = action_index  # The move (column) that led here.
        self.c = 1  # Exploration constant.
        self.reward = 0.0
        self.starting_player = starting_player

    def getUCTscore(self, parent_log=None):
        """
        Calculate the UCT score for this node.
        Uses precomputed parent_log if available.
        """
        if self.visits == 0:
            return float('inf')
        if parent_log is None:
            parent_visits = self.parent.visits if self.parent else 1
            parent_log = math.log(parent_visits)
        # Apply center bias if action is defined.
        bias = center_bias(self.game_state, self.action_index) if self.action_index is not None else 0.0
        return (self.node_value / self.visits) + bias + self.c * math.sqrt(parent_log / self.visits)

    def create_child_nodes(self):
        """
        Expand this node by creating a child for each valid move.
        """
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=2, min_rollouts=50000000, min_time=0.0, max_time=5.5, batch_size=8):
        """
        Explore the tree using parallel rollouts.
        Rollouts are scheduled in batches.
        """
        start_time = time.perf_counter()
        rollouts = 0

        # Cache local functions for speed.
        rand_choice = random.choice
        get_time = time.perf_counter
        batch = []  # List of tuples: (future, node)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                elapsed = get_time() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    break

                current = self
                # --- Selection Phase ---
                while current.child_nodes:
                    # Compute parent's log once for all children.
                    parent_visits = current.visits if current.visits else 1
                    parent_log = math.log(parent_visits)
                    best_score = -float('inf')
                    best_children = []
                    for child in current.child_nodes.values():
                        score = child.getUCTscore(parent_log)
                        if score > best_score:
                            best_children = [child]
                            best_score = score
                        elif score == best_score:
                            best_children.append(child)
                    current = rand_choice(best_children)

                # --- Expansion Phase ---
                if not current.done:
                    current.create_child_nodes()
                    if current.child_nodes:
                        current = rand_choice(list(current.child_nodes.values()))

                # Schedule a rollout in parallel.
                future = executor.submit(rollout_simulation, current.game_state, minimax_depth)
                batch.append((future, current))
                rollouts += 1

                # Process batch when full.
                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()
                        except Exception:
                            reward = 0.0
                        # --- Backpropagation Phase ---
                        node_to_update = node
                        while node_to_update:
                            node_to_update.visits += 1
                            node_to_update.node_value += reward
                            node_to_update = node_to_update.parent
                    batch.clear()

            # Process any remaining rollouts.
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
        Perform a rollout from the current state with center bias used during move selection.
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

        while not gameIsOver(board_state):
            valid_moves = board_state.get_valid_moves()
            if not valid_moves:
                return 0.5  # Draw
            # Compute center biases for a weighted random choice.
            biases = [center_bias(board_state, move) for move in valid_moves]
            action = random.choices(valid_moves, weights=biases, k=1)[0]
            board_state.current_player = current_player
            board_state.play_move(action)
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(board_state, AI_PLAYER)

    def next(self):
        """
        Return the child with the highest visit count.
        """
        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children available. Ensure exploration has been performed.")
        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """
        Move to the node corresponding to the given player move.
        """
        if playerMove in self.child_nodes:
            new_root = self.child_nodes[playerMove]
        else:
            new_board = copy_board(self.game_state)
            new_board.play_move(playerMove)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root
        return new_root
