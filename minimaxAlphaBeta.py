from copy import deepcopy
from random import shuffle

# Assume these are defined in your adapted utility_functions and bitboard modules:
from utility_functions import utilityValue, gameIsOver, HUMAN_PLAYER, AI_PLAYER


# We assume that your BitBoard class replaces the old board module.
# from board import getValidMoves, makeMove  --> replaced by BitBoard methods

def MiniMaxAlphaBeta(bitboard, depth, player):
    """
    Return the best move (column index) and its score using MiniMax with Alpha-Beta pruning.

    Parameters:
      - bitboard: current BitBoard state
      - depth: search depth
      - player: the maximizing player's symbol (e.g., HUMAN_PLAYER or AI_PLAYER)
    """
    valid_moves = bitboard.get_valid_moves()  # List of valid column moves
    shuffle(valid_moves)  # Randomize move order

    best_move = None
    best_score = float("-inf")

    alpha = float("-inf")
    beta = float("inf")
    opponent = HUMAN_PLAYER if player == AI_PLAYER else AI_PLAYER

    for move in valid_moves:
        temp_board = deepcopy(bitboard)
        # Force the move to be played by the maximizing player.
        temp_board.current_player = player
        _, _, _ = temp_board.play_move(move)
        board_score = minimizeBeta(temp_board, depth - 1, alpha, beta, player, opponent)

        if board_score > best_score:
            best_score = board_score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            break  # Alpha-beta cutoff

    return best_move, best_score


def minimizeBeta(bitboard, depth, alpha, beta, player, opponent):
    """
    Minimizing part of MiniMax. Simulate opponent moves.

    Parameters:
      - bitboard: current BitBoard state
      - depth: search depth
      - alpha: current alpha value
      - beta: current beta value
      - player: maximizing player's symbol
      - opponent: minimizing player's symbol
    """
    valid_moves = bitboard.get_valid_moves()
    if depth == 0 or not valid_moves or gameIsOver(bitboard):
        return utilityValue(bitboard, player)

    for move in valid_moves:
        temp_board = deepcopy(bitboard)
        # Force the move to be played by the opponent.
        temp_board.current_player = opponent
        _, _, _ = temp_board.play_move(move)
        board_score = maximizeAlpha(temp_board, depth - 1, alpha, beta, player, opponent)

        beta = min(beta, board_score)
        if alpha >= beta:
            break  # Pruning

    return beta


def maximizeAlpha(bitboard, depth, alpha, beta, player, opponent):
    """
    Maximizing part of MiniMax. Simulate moves for the maximizing player.

    Parameters:
      - bitboard: current BitBoard state
      - depth: search depth
      - alpha: current alpha value
      - beta: current beta value
      - player: maximizing player's symbol
      - opponent: minimizing player's symbol
    """
    valid_moves = bitboard.get_valid_moves()
    if depth == 0 or not valid_moves or gameIsOver(bitboard):
        return utilityValue(bitboard, player)

    for move in valid_moves:
        temp_board = deepcopy(bitboard)
        # Force the move to be played by the maximizing player.
        temp_board.current_player = player
        _, _, _ = temp_board.play_move(move)
        board_score = minimizeBeta(temp_board, depth - 1, alpha, beta, player, opponent)

        alpha = max(alpha, board_score)
        if alpha >= beta:
            break  # Pruning

    return alpha
