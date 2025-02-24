import concurrent.futures
import math
import os
import threading
import time
from random import choice

from utility_functions import AI_PLAYER, HUMAN_PLAYER, EndValue, gameIsOver, countSequence

# Constants
C_PUCT = 1.4142         # Exploration constant
MAX_ROLLOUTS = 200000
MAX_TIME = 10.0
CENTER_BIAS = 0.2       # Bias favoring center moves
DEPTH_DISCOUNT = 0.9    # Favor shorter win paths
VIRTUAL_LOSS = 1        # Constant value for virtual loss

class EnhancedNode:
    def __init__(self, game_state, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0  # Heuristic prior probability

        # Progressive bias parameters
        self.min_depth = float('inf')
        self.max_depth = 0

    def is_leaf(self):
        return not self.children

    def select_child(self):
        """Selects the best child based on value, exploration, and biases."""
        log_visits = math.log(self.visits) if self.visits > 0 else 0

        def child_score(child):
            depth_bias = DEPTH_DISCOUNT ** child.min_depth
            exploit = child.value / (child.visits + 1e-7)
            explore = C_PUCT * child.prior * math.sqrt(log_visits) / (child.visits + 1)
            center_bonus = self._center_bias(child.action) if isinstance(child.action, (int, float)) else 0
            return exploit + (explore * depth_bias) + center_bonus

        return max(self.children.values(), key=child_score)

    def expand(self):
        """
        Expands one promising child rather than all moves at once.
        This lazy expansion saves resources.
        """
        if not self.children:
            valid_moves = self.game_state.get_valid_moves()
            # Choose the move with the highest heuristic prior
            best_move = max(valid_moves, key=lambda move: self._calculate_prior_for_move(move))
            new_state = self.game_state.copy()
            new_state.play_move(best_move)
            child = EnhancedNode(new_state, self, best_move)
            child.prior = self._calculate_prior(new_state)
            self.children[best_move] = child

    def _calculate_prior_for_move(self, move):
        """Calculate prior for a specific move by simulating it."""
        new_state = self.game_state.copy()
        new_state.play_move(move)
        return self._calculate_prior(new_state)

    def _calculate_prior(self, new_state):
        """Combines threat detection and positional evaluation to set a heuristic prior."""
        # Immediate win/loss detection
        if countSequence(new_state, AI_PLAYER, 4) > 0:
            return float('inf')
        if countSequence(new_state, HUMAN_PLAYER, 4) > 0:
            return -float('inf')

        prior = 0.0
        # Threat analysis: weigh sequences of different lengths
        prior += 0.5 * countSequence(new_state, AI_PLAYER, 3)
        prior -= 0.7 * countSequence(new_state, HUMAN_PLAYER, 3)
        prior += 0.2 * countSequence(new_state, AI_PLAYER, 2)
        prior -= 0.3 * countSequence(new_state, HUMAN_PLAYER, 2)

        return prior

    def _center_bias(self, column):
        """Encourages moves toward the center (assumes columns 0-6 with center at 3)."""
        return CENTER_BIAS * (1 - abs(column - 3) / 3)

    def backpropagate(self, value, depth):
        """Backpropagates the simulation result using a negamax style update."""
        self.visits += 1
        self.value += value
        self.min_depth = min(self.min_depth, depth)
        self.max_depth = max(self.max_depth, depth)

        if self.parent:
            # Negamax propagation with depth discounting
            self.parent.backpropagate(-value * DEPTH_DISCOUNT, depth + 1)

class ParallelMCTS:
    def __init__(self, root_state, max_time=MAX_TIME, max_rollouts=MAX_ROLLOUTS):
        self.root = EnhancedNode(root_state)
        self.max_time = max_time
        self.max_rollouts = max_rollouts
        self.batch_size = min(4, os.cpu_count() or 1)
        self.lock = threading.Lock()  # For thread safety
        self.virtual_loss = VIRTUAL_LOSS

    def search(self):
        start_time = time.time()
        rollouts = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            try:
                while (time.time() - start_time < self.max_time and
                       rollouts < self.max_rollouts):

                    batch_paths = []
                    for _ in range(self.batch_size):
                        path = self._select_path()
                        if self._valid_path(path):
                            # Apply virtual loss before simulation
                            self._apply_virtual_loss(path, revert=False)
                            batch_paths.append(path)
                            rollouts += 1

                    batch_futures = [executor.submit(self._simulate, p) for p in batch_paths]
                    futures.extend(batch_futures)

                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures.copy()):
                        try:
                            result = future.result()
                            self._process_result(result)
                        except Exception as e:
                            print(f"Ignoring failed future: {str(e)}")
                        finally:
                            with self.lock:
                                if future in futures:
                                    futures.remove(future)
            except Exception as e:
                print(f"Search terminated: {str(e)}")
        return self.root

    def _valid_path(self, path):
        """Checks if the selected path is valid for simulation."""
        if not path:
            return False
        try:
            node = path[-1]
            return node.game_state is not None and not node.game_state.is_terminal()
        except Exception:
            return False

    def _apply_virtual_loss(self, path, revert=False):
        """Applies or reverts virtual loss along the simulation path."""
        for node in path:
            if revert:
                node.visits -= self.virtual_loss
                node.value += self.virtual_loss  # Revert the loss
            else:
                node.visits += self.virtual_loss
                node.value -= self.virtual_loss

    def _process_result(self, result):
        """Processes the simulation result in a thread-safe manner."""
        with self.lock:
            if result.get('node'):
                # Revert virtual loss before backpropagation
                self._apply_virtual_loss(result['path'], revert=True)
                result['node'].backpropagate(result['value'], result['depth'])

    def best_move(self):
        """
        Returns the best move using a combination of average value and visit count.
        Falls back to center bias if no child is sufficiently visited.
        """
        try:
            if self.root.children:
                best_action = max(
                    self.root.children.values(),
                    key=lambda child: (child.value / (child.visits + 1e-7)) + 0.1 * child.visits
                )
                return best_action.action

            valid_moves = self.root.game_state.get_valid_moves()
            if valid_moves:
                return min(valid_moves, key=lambda x: abs(x - 3))
            return 3  # Fallback to center column
        except Exception as e:
            print(f"Move selection failed: {str(e)}")
            return 3

    def _simulate(self, path):
        """
        Expands the leaf node, selects a promising child (if possible),
        then performs a full rollout simulation until terminal state.
        """
        leaf = path[-1]

        if not leaf.game_state.is_terminal():
            leaf.expand()
            if leaf.children:
                node = leaf.select_child()
                path.append(node)
            else:
                node = leaf
        else:
            node = leaf

        # Perform a full simulation (rollout) until terminal state
        value = self._simulate_rollout(node.game_state)
        return {'path': path, 'node': node, 'value': value, 'depth': len(path)}

    def _simulate_rollout(self, state):
        """
        Plays out a full random simulation (rollout) until the terminal state is reached.
        Returns the evaluation using EndValue (e.g., +1, -1, 0).
        """
        rollout_state = state.copy()
        while not rollout_state.is_terminal():
            valid_moves = rollout_state.get_valid_moves()
            move = choice(valid_moves)
            rollout_state.play_move(move)
        return EndValue(rollout_state)

    def _select_path(self):
        """Selects a path from the root to a leaf node for simulation."""
        path = []
        current = self.root
        while True:
            path.append(current)
            if current.is_leaf() or current.game_state.is_terminal():
                break
            current = current.select_child()
        return path

# Usage example
def get_ai_move(game_state):
    mcts = ParallelMCTS(game_state)
    mcts.search()
    return mcts.best_move()
