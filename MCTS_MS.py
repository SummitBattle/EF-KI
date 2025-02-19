import random
import time
import math
from copy import deepcopy
import concurrent.futures

from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue

import logging

# Logging-Konfiguration
logging.basicConfig(
    filename='rollouts.log',       # Log-Datei
    format='%(asctime)s - %(message)s',
    level=logging.INFO             # Kann auf DEBUG geändert werden für detailliertere Logs
)


def copy_board(bitboard):
    """Schneller Board-Klon: Nutzt eine eingebaute copy()-Methode, falls vorhanden."""
    if hasattr(bitboard, 'copy'):
        return bitboard.copy()
    return deepcopy(bitboard)


def count_moves(bitboard):
    """
    Zählt die Züge für jeden Spieler anhand der Bitboard-Repräsentation.
    Annahme:
      - bitboard.board1 enthält die Züge des Menschen ('x')
      - bitboard.board2 enthält die Züge der KI ('o')
    """
    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")
    return ai_moves, human_moves


def get_move_column(move):
    """
    Extrahiert den Spaltenindex aus einem Zug.
    Bei einem Tupel wird das zweite Element (Spalte) zurückgegeben.
    Bei einem int wird dieser direkt zurückgegeben.
    """
    if isinstance(move, int):
        return move
    elif isinstance(move, tuple):
        return move[1]
    return move


def biased_random_move(valid_moves, center_col=3, bias_strength=0.09):
    """
    Wählt einen Zug mit leichtem Center-Bias.
    Für ein Standard-Connect-4-Brett (7 Spalten) ist die mittlere Spalte 3.
    Züge, die näher am Zentrum liegen, erhalten ein höheres Gewicht.
    """
    weights = []
    max_distance = 3  # Maximale Distanz in einem 7-Spalten-Brett
    for move in valid_moves:
        col = get_move_column(move)
        weight = 1 + bias_strength * (max_distance - abs(col - center_col))
        weights.append(weight)
    return random.choices(valid_moves, weights=weights, k=1)[0]


def rollout_simulation(game_state, minimax_depth):
    """
    Führt eine Rollout-Simulation aus einem gegebenen Spielzustand durch.
    Diese Funktion kann parallel ausgeführt werden.
    """
    board_state = copy_board(game_state)

    # Falls das Spiel bereits vorbei ist, sofort den Endwert zurückgeben
    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    # --- Erster Zug mittels Minimax ---
    best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    # Wechsel des aktiven Spielers für die anschließende Random-Rollout-Phase.
    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

    # --- Random-Rollout-Phase mit Center-Bias ---
    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5  # Unentschieden, falls keine Züge verfügbar sind.
        action = biased_random_move(valid_moves)
        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)


class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'starting_player')

    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # Dictionary: action -> Node
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # Instanz des BitBoard
        self.done = done
        self.action_index = action_index  # Der Zug, der zu diesem Node geführt hat (z. B. (row, col))
        self.c = 1.2  # Explorationskonstante
        self.starting_player = starting_player

    def getUCTscore(self, center_col=3, bias_strength=0.01):
        """
        Berechnet den UCT-Score dieses Knotens. Zusätzlich wird ein Bias für Züge in der Brettmitte addiert.
        """
        if self.visits == 0:
            return float('inf')
        # Berechne logarithmischer Anteil des Eltern-Knotens (schneller als wiederholtes Berechnen)
        parent_visits = self.parent.visits if self.parent else 1
        log_parent = math.log(parent_visits) if parent_visits > 0 else 0
        exploitation = self.node_value / self.visits
        exploration = self.c * math.sqrt(log_parent / self.visits)
        uct_score = exploitation + exploration

        # Center-Bias hinzufügen
        move_col = get_move_column(self.action_index)
        max_distance = 3
        center_bias = bias_strength * (max_distance - abs(move_col - center_col))
        return uct_score + center_bias

    def create_child_nodes(self):
        """
        Erweitert den aktuellen Node um alle möglichen Kind-Knoten, basierend auf den gültigen Zügen.
        """
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=2, min_rollouts=50000000, min_time=0.0, max_time=6.0, batch_size=32):
        """
        Erforscht den Baum mithilfe paralleler Rollouts.
        Statt jeden Rollout synchron auszuführen, werden Batches von Rollouts parallel geplant.
        """
        start_time = time.perf_counter()
        rollouts = 0
        rand_choice = random.choice
        get_time = time.perf_counter
        batch = []  # Liste von Tupeln: (future, node)

        # Verwende ThreadPoolExecutor für leichte Parallelisierung
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                elapsed = get_time() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    break

                current = self
                # --- Selektion ---
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

                # Plane die Rollout-Simulation parallel ein.
                future = executor.submit(rollout_simulation, current.game_state, minimax_depth)
                batch.append((future, current))
                rollouts += 1

                # Verarbeite den Batch, sobald die festgelegte Batch-Größe erreicht ist.
                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()
                        except Exception:
                            reward = 0.0  # Fallback, falls die Simulation fehlschlägt.
                        # --- Backpropagation ---
                        node_to_update = node
                        while node_to_update:
                            node_to_update.visits += 1
                            node_to_update.node_value += reward
                            node_to_update = node_to_update.parent
                    batch.clear()

            # Verarbeite verbleibende Futures im Batch.
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

        logging.info(f"Anzahl der Rollouts: {rollouts}")
        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """
        Führt einen Rollout durch, bei dem beim Zugauswahl ein Center-Bias angewandt wird.
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
                return 0.5  # Unentschieden, falls keine Züge verfügbar
            action = biased_random_move(valid_moves)
            board_state.current_player = current_player
            board_state.play_move(action)
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(board_state, AI_PLAYER)

    def next(self):
        """
        Wählt den besten nächsten Zug basierend auf der höchsten Besuchszahl.
        """
        if self.done:
            raise ValueError("Spiel beendet. Kein nächster Zug verfügbar.")
        if not self.child_nodes:
            raise ValueError("Keine Kind-Knoten vorhanden. Führe zuerst eine Exploration durch.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)
        best_child.game_state.print_board()
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """
        Aktualisiert den Baum anhand des Spielerzuges.
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
