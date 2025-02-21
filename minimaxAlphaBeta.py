from random import shuffle
from board import BitBoard  # Assuming your BitBoard class is correctly implemented
from utility_functions import utilityValue, gameIsOver, HUMAN_PLAYER, AI_PLAYER
from copy import deepcopy


def minimax_alpha_beta(bitboard, depth, alpha, beta, maximizingPlayer):
    """
    Perform Minimax with Alpha-Beta Pruning on a BitBoard representation of Connect 4.

    Parameters:
        - bitboard: The current BitBoard state.
        - depth: Remaining depth for the search.
        - alpha: Best already explored option along the path to maximizer.
        - beta: Best already explored option along the path to minimizer.
        - maximizingPlayer: True if it's the AI's turn, False if it's the opponent's turn.

    Returns:
        - best_move: Column index of the best move.
        - best_score: Score of the best move.
    """
    valid_moves = bitboard.get_valid_moves()
    shuffle(valid_moves)  # Randomize move order to improve AI variety

    if depth == 0 or gameIsOver(bitboard) or not valid_moves:
        return None, utilityValue(bitboard, AI_PLAYER)  # Evaluate board state

    best_move = None

    if maximizingPlayer:  # AI's turn (maximize)
        best_score = float("-inf")
        for move in valid_moves:
            temp_board = deepcopy(bitboard)  # Create a copy of the bitboard
            _, _, win = temp_board.play_move(move)  # Simulate move

            if win:  # If AI can win immediately, return move
                return move, float("inf")

            _, board_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False)

            if board_score > best_score:
                best_score = board_score
                best_move = move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Alpha-Beta Pruning

    else:  # Opponent's turn (minimize)
        best_score = float("inf")
        for move in valid_moves:
            temp_board = deepcopy(bitboard)
            _, _, win = temp_board.play_move(move)

            if win:  # If opponent can win immediately, return worst-case score
                return move, float("-inf")

            _, board_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True)

            if board_score < best_score:
                best_score = board_score
                best_move = move

            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha-Beta Pruning

    return best_move, best_score
