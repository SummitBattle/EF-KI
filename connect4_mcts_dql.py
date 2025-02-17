import os
import random
import time
import math
from copy import deepcopy
import concurrent.futures
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------
# Console Colors & Board Setup
# ----------------------------
RED     = '\033[1;31;40m'
BLUE    = '\033[1;34;40m'
YELLOW  = '\033[1;33;40m'
WHITE   = '\033[1;37;40m'

BOARD_WIDTH  = 7
BOARD_HEIGHT = 6
COL_SIZE = BOARD_HEIGHT + 1  # We use an extra bit per column as a sentinel.

# Precompute masks for each column.
bottom_mask = [1 << (col * COL_SIZE) for col in range(BOARD_WIDTH)]
column_mask = [(((1 << BOARD_HEIGHT) - 1) << (col * COL_SIZE)) for col in range(BOARD_WIDTH)]
top_mask = [1 << (col * COL_SIZE + BOARD_HEIGHT - 1) for col in range(BOARD_WIDTH)]

FULL_MASK = 0
for col in range(BOARD_WIDTH):
    FULL_MASK |= column_mask[col]

def is_winning_position(pos):
    """
    Check whether the given bitboard 'pos' contains a winning 4‑in‑a‑row.
    Vertical neighbors are 1 bit apart.
    Horizontal neighbors are COL_SIZE (7) bits apart.
    Diagonals use shifts of (COL_SIZE - 1)=6 and (COL_SIZE + 1)=8.
    """
    # Vertical
    m = pos & (pos >> 1)
    if m & (m >> 2):
        return True
    # Horizontal
    m = pos & (pos >> COL_SIZE)
    if m & (m >> (2 * COL_SIZE)):
        return True
    # Diagonal (/)
    m = pos & (pos >> (COL_SIZE - 1))
    if m & (m >> (2 * (COL_SIZE - 1))):
        return True
    # Diagonal (\)
    m = pos & (pos >> (COL_SIZE + 1))
    if m & (m >> (2 * (COL_SIZE + 1))):
        return True
    return False

# ----------------------------
# BitBoard Class Definition
# ----------------------------
class BitBoard:
    def __init__(self):
        self.board1 = 0  # Bits for Player 1 (AI; represented with 'x')
        self.board2 = 0  # Bits for Player 2 (opponent; represented with 'o')
        self.mask = 0    # Overall occupancy (board1 OR board2)
        self.current_player = 1  # Player 1 starts (our AI)
        self.moves = []  # History of moves (columns played)

    def can_play(self, col):
        """Return True if there is room in the given column (0-indexed)."""
        return (self.mask & top_mask[col]) == 0

    def get_valid_moves(self):
        """Return a list of valid columns (0-indexed) where a move can be made."""
        return [col for col in range(BOARD_WIDTH) if self.can_play(col)]

    def play_move(self, col):
        """
        Play a move in the given column.
        Returns (row, col, win_flag).
        """
        if not self.can_play(col):
            raise ValueError(f"Column {col+1} is full")
        # Compute the bit corresponding to the lowest empty cell.
        move = (self.mask + bottom_mask[col]) & column_mask[col]
        if self.current_player == 1:
            self.board1 |= move
        else:
            self.board2 |= move
        self.mask |= move
        self.moves.append(col)
        # Compute row: bit position offset within the column.
        row = (move.bit_length() - 1) - (col * COL_SIZE)
        # Check win condition.
        if self.current_player == 1:
            win = is_winning_position(self.board1)
        else:
            win = is_winning_position(self.board2)
        # Switch player.
        self.current_player = 2 if self.current_player == 1 else 1
        return row, col, win

    def is_board_filled(self):
        """Return True if the board is completely filled."""
        return self.mask == FULL_MASK

    def print_board(self):
        """Clear the screen and print the board in a human‑readable form."""
        os.system('cls' if os.name == 'nt' else 'clear')
        moves_played = len(self.moves)
        print('')
        print(YELLOW + '         ROUND #' + str(moves_played) + WHITE)
        print('')
        print("\t      1   2   3   4   5   6   7")
        print("\t      -   -   -   -   -   -   -")
        for r in range(BOARD_HEIGHT - 1, -1, -1):
            print(WHITE + "\t", r+1, ' ', end="")
            for col in range(BOARD_WIDTH):
                bit = 1 << (col * COL_SIZE + r)
                if self.board1 & bit:
                    piece = BLUE + 'x' + WHITE
                elif self.board2 & bit:
                    piece = RED + 'o' + WHITE
                else:
                    piece = ' '
                print("| " + piece, end=" ")
            print("|")
        print('')

    def get_board_array(self):
        """
        Convert the bitboard into a 2D numpy array.
        Uses:
          0 for an empty cell,
          1 for Player 1,
          2 for Player 2.
        """
        board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        for col in range(BOARD_WIDTH):
            for r in range(BOARD_HEIGHT):
                bit = 1 << (col * COL_SIZE + r)
                if self.board1 & bit:
                    board[r, col] = 1
                elif self.board2 & bit:
                    board[r, col] = 2
        return board

# ----------------------------
# Utility Functions & Constants
# ----------------------------
# For our purposes, we define:
AI_PLAYER = 1
HUMAN_PLAYER = 2

def EndValue(game_state, ai_player):
    """
    Compute a reward from the perspective of AI_PLAYER.
    Win: 1.0, Loss: 0.0, Draw: 0.5.
    """
    if game_state.is_board_filled():
        return 0.5
    if ai_player == 1:
        return 1.0 if is_winning_position(game_state.board1) else 0.0
    else:
        return 1.0 if is_winning_position(game_state.board2) else 0.0

def gameIsOver(game_state):
    """Return True if the game is over (win or full board)."""
    return (game_state.is_board_filled() or
            is_winning_position(game_state.board1) or
            is_winning_position(game_state.board2))

def copy_board(bitboard):
    """Return a deep copy of the board."""
    if hasattr(bitboard, 'copy'):
        return bitboard.copy()
    return deepcopy(bitboard)

def get_move_column(move):
    """Helper: extract column index from move."""
    if isinstance(move, int):
        return move
    elif isinstance(move, tuple):
        return move[1]
    return move

def biased_random_move(valid_moves, center_col=3, bias_strength=0.01):
    """
    Select a move with a slight center bias.
    Moves closer to the center (col 3) get a higher weight.
    """
    weights = []
    max_distance = 3
    for move in valid_moves:
        col = get_move_column(move)
        weight = 1 + bias_strength * (max_distance - abs(col - center_col))
        weights.append(weight)
    return random.choices(valid_moves, weights=weights, k=1)[0]

# ----------------------------
# Deep Q‑Learning Components
# ----------------------------
class QNetwork(nn.Module):
    """
    A simple fully connected Q‑Network.
    Adjust input_dim and output_dim based on your board representation.
    """
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def board_to_tensor(game_state):
    """
    Convert the BitBoard state to a tensor for the Q‑network.
    """
    board_array = game_state.get_board_array()
    tensor = torch.FloatTensor(board_array).flatten().unsqueeze(0)  # shape: (1, num_features)
    return tensor

def rollout_simulation_with_q(game_state, minimax_depth, q_network, epsilon=0.1):
    """
    Perform a rollout from game_state.
    First, use a minimax move; then rollout with an epsilon‑greedy policy guided by the Q‑network.
    """
    board_state = copy_board(game_state)
    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    # --- First Move with Minimax (stub) ---
    best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    # Switch current player for the rollout phase.
    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

    # --- Rollout Phase with Deep Q‑Learning Guidance ---
    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5  # Draw

        # Epsilon‑greedy: choose random move with probability epsilon.
        if random.random() < epsilon:
            action = biased_random_move(valid_moves)
        else:
            q_values = []
            for move in valid_moves:
                new_board = copy_board(board_state)
                new_board.play_move(move)
                state_tensor = board_to_tensor(new_board)
                with torch.no_grad():
                    q_val = q_network(state_tensor)
                q_values.append(q_val.item())
            best_index = np.argmax(q_values)
            action = valid_moves[best_index]

        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)

# ----------------------------
# MCTS Node with Deep Q‑Learning Integration
# ----------------------------
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
        self.action_index = action_index  # The move that led to this node
        self.c = 1.41  # Exploration constant
        self.reward = 0.0
        self.starting_player = starting_player

    def getUCTscore(self, center_col=3, bias_strength=0.01):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        uct_score = (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)
        if isinstance(self.action_index, tuple):
            move_col = self.action_index[1]
        else:
            move_col = self.action_index
        max_distance = 3
        center_bias = bias_strength * (max_distance - abs(move_col - center_col))
        return uct_score + center_bias

    def create_child_nodes(self):
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=2, min_rollouts=1000, min_time=0.0, max_time=1.0,
                batch_size=4, q_network=None, epsilon=0.1):
        """
        Explore the tree using parallel rollouts.
        If a q_network is provided, it uses rollout_simulation_with_q.
        """
        start_time = time.perf_counter()
        rollouts = 0
        rand_choice = random.choice
        get_time = time.perf_counter
        batch = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                elapsed = get_time() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    break

                current = self
                # --- Selection ---
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

                # --- Expansion ---
                if not current.done:
                    current.create_child_nodes()
                    if current.child_nodes:
                        current = rand_choice(list(current.child_nodes.values()))

                # --- Rollout ---
                if q_network is not None:
                    future = executor.submit(rollout_simulation_with_q, current.game_state, minimax_depth, q_network, epsilon)
                else:
                    # Fallback if no Q-network is provided.
                    future = executor.submit(rollout_simulation_with_q, current.game_state, minimax_depth, q_network, epsilon)
                batch.append((future, current))
                rollouts += 1

                if len(batch) >= batch_size:
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

        logging.info(f"Number of rollouts: {rollouts}")
        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """Fallback rollout (without Q-network guidance)."""
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
                return 0.5
            action = biased_random_move(valid_moves)
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

# ----------------------------
# MiniMaxAlphaBeta Stub (for demonstration)
# ----------------------------
def MiniMaxAlphaBeta(game_state, depth, current_player):
    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return None, 0
    # For this stub, we choose a random move.
    move = random.choice(valid_moves)
    return move, 0

# ----------------------------
# Replay Buffer for Q-Network Training
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    filename='rollouts.log',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

# ----------------------------
# Main Training Loop (Self‑Play)
# ----------------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize a Q-network.
    input_dim = BOARD_HEIGHT * BOARD_WIDTH  # Flattened board array.
    output_dim = 1  # Scalar value for state evaluation.
    q_net = QNetwork(input_dim, output_dim)
    q_optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Initialize replay buffer.
    replay_buffer = ReplayBuffer(capacity=1000)

    num_episodes = 10  # Adjust number of self‑play episodes as desired.
    batch_size = 4

    for episode in range(num_episodes):
        board = BitBoard()
        episode_states = []  # To store states when AI (player 1) is about to move.

        # Play one episode (self‑play).
        while not gameIsOver(board):
            state_tensor = board_to_tensor(board)
            if board.current_player == AI_PLAYER:
                # Use MCTS (with Q-network guidance) to select move.
                root = Node(board, False, None, None, board.current_player)
                # Explore for a short time (e.g., 1 second).
                root.explore(minimax_depth=2, max_time=1.0, q_network=q_net, epsilon=0.1)
                try:
                    best_node, best_move = root.next()
                    action = best_move
                except ValueError:
                    # Fallback in case exploration did not produce a move.
                    action = random.choice(board.get_valid_moves())
                # Record state for training.
                episode_states.append(state_tensor)
            else:
                # Opponent plays randomly.
                action = random.choice(board.get_valid_moves())

            board.play_move(action)

        # Determine the outcome from AI's perspective.
        outcome = EndValue(board, AI_PLAYER)
        print(f"Episode {episode+1}: Outcome (reward) = {outcome}")

        # Add experiences from AI moves to the replay buffer.
        for state in episode_states:
            replay_buffer.add((state, outcome))

        # Train the Q-network if enough samples are available.
        if len(replay_buffer.buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states = torch.cat([transition[0] for transition in batch], dim=0)
            targets = torch.FloatTensor([transition[1] for transition in batch]).unsqueeze(1)
            predictions = q_net(states)
            loss = loss_fn(predictions, targets)
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()
            print(f"Episode {episode+1}: Training Loss = {loss.item():.4f}")
        else:
            print(f"Episode {episode+1}: Not enough samples for training yet.")

    # After training, demonstrate one move from a new board.
    print("\nDemonstration of move selection after training:")
    demo_board = BitBoard()
    demo_board.print_board()
    root = Node(demo_board, False, None, None, demo_board.current_player)
    root.explore(minimax_depth=2, max_time=1.0, q_network=q_net, epsilon=0.1)
    best_node, best_move = root.next()
    print(f"Best move selected: Column {best_move + 1}")
    best_node.game_state.print_board()
