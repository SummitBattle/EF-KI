import random
import time
import logging
import matplotlib.pyplot as plt
import networkx as nx

from board import *
from copy import deepcopy

from utility_functions import utilityValue, gameIsOver, AI_PLAYER, HUMAN_PLAYER, countSequence


class Node:
    def __init__(self, game_state, done, parent_node, action_index, reward):
        self.parent = parent_node
        self.child_nodes = {}
        self.visits = 0
        self.node_value = 0  # Accumulated value from simulations
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 1.7  # Exploration constant
        self.reward = reward

    def getUCTscore(self):
        """Calculate the UCT score for this node."""
        if self.visits == 0:
            return float('inf')  # If the node has not been visited, prioritize exploration

        parent_visits = max(1, self.parent.visits) if self.parent else 1  # Prevent division by zero

        # UCT formula: Exploitation + Exploration
        uct_score = (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)
        return uct_score

    def create_child_nodes(self):
        valid_moves = getValidMoves(self.game_state)
        for action in valid_moves:
            result = makeMove(deepcopy(self.game_state), action, AI_PLAYER)
            child_board_state = result[0]

            done = gameIsOver(child_board_state)
            reward = EndValue(child_board_state, AI_PLAYER)
            self.child_nodes[action] = Node(child_board_state, done, self, action, reward)

    def explore(self, minimax_depth=3, min_rollouts=200, min_time=0.0):
        """Select the best child node based on UCT or expand a new node if possible."""
        start_time = time.time()
        rollouts = 0

        while rollouts < min_rollouts or (time.time() - start_time) < min_time:
            current = self

            # Traverse the tree until a leaf node is reached
            while current.child_nodes:
                child_scores = {a: c.getUCTscore() for a, c in current.child_nodes.items()}
                max_U = max(child_scores.values())
                best_actions = [a for a, score in child_scores.items() if score == max_U]

                action = random.choice(best_actions)
                current = current.child_nodes[action]

            # Perform rollouts from this leaf node
            if not current.child_nodes:
                current.create_child_nodes()
                if current.child_nodes:
                    current = random.choice(list(current.child_nodes.values()))

            current.reward = self.rollout(minimax_depth)

            rollouts += 1

            # Backpropagate the reward
            parent = current
            while parent:
                parent.visits += 1
                parent.node_value += current.reward
                logging.debug(f'Backpropagating Reward: {current.reward} to Parent: {id(parent)}')
                parent = parent.parent

        return self

    def rollout(self, minimax_depth):
        new_board = deepcopy(self.game_state)
        current_player = AI_PLAYER

        while not gameIsOver(new_board):
            valid_moves = getValidMoves(new_board)

            if not valid_moves:
                return 0.5  # Draw (no valid moves left)

            # Use a random move
            action = random.choice(valid_moves)

            # Apply the chosen move
            new_board, _, _ = makeMove(new_board, action, current_player)

            # Check if the game is over
            if gameIsOver(new_board):
                return EndValue(new_board, AI_PLAYER)  # Score from AI's perspective

            # Switch players
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(new_board, AI_PLAYER)

    def next(self):
        """Return the best child based on visit count."""
        if self.done:
            raise ValueError("Game has ended")

        if not self.child_nodes:
            raise ValueError("No children found and the game hasn't ended")

        max_visits = max(node.visits for node in self.child_nodes.values())
        best_children = [c for c in self.child_nodes.values() if c.visits == max_visits]

        best_child = random.choice(best_children)
        return best_child, best_child.action_index

    def visualize(self, max_depth=3, min_visits=0):
        """Visualize the current state of the MCTS tree up to a max depth and min visits."""
        G = nx.DiGraph()
        self._add_node_to_graph(G, self, depth=0, max_depth=max_depth, min_visits=min_visits)

        pos = self.hierarchy_pos(G, id(self))

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_size=700, node_color='skyblue', edge_color='gray')

        # Create labels for visits (V) and node value (NV)
        labels = {node: f'V: {data["visits"]}\nNV: {data["node_value"]:.2f}' for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

        plt.title("MCTS Tree Visualization")
        plt.show()

    def _add_node_to_graph(self, G, node, depth, max_depth, min_visits):
        """Recursively add nodes and edges to the graph, limiting by depth and visits."""
        if depth > max_depth or node.visits < min_visits:
            return

        G.add_node(id(node), visits=node.visits, node_value=node.node_value)
        if node.parent:
            G.add_edge(id(node.parent), id(node))

        for child in node.child_nodes.values():
            self._add_node_to_graph(G, child, depth + 1, max_depth, min_visits)

    def hierarchy_pos(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        If there is a cycle that is reachable from root, then this will see infinite recursion.
        G: the graph (must be a tree)
        root: the root node of current branch
        width: horizontal space allocated for this branch - avoids overlap with other branches
        vert_gap: gap between levels of hierarchy
        vert_loc: vertical location of root
        xcenter: horizontal location of root
        """
        pos = {root: (xcenter, vert_loc)}
        neighbors = list(G.neighbors(root))
        if len(neighbors) != 0:
            dx = width / len(neighbors)
            nextx = xcenter - width / 2 - dx / 2
            for neighbor in neighbors:
                nextx += dx
                pos.update(self.hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx))
        return pos