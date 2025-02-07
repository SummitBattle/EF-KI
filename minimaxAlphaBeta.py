from board import *
from random import shuffle
from copy import deepcopy

def MiniMaxAlphaBeta(board, depth, player):
    valid_moves = getValidMoves(board)
    if not valid_moves:
        return None  # No moves possible

    shuffle(valid_moves)  # Add randomness to move selection
    best_move = valid_moves[0]
    best_score = float("-inf")

    alpha = float("-inf")
    beta = float("inf")

    opponent = HUMAN_PLAYER if player == AI_PLAYER else AI_PLAYER

    for move in valid_moves:
        temp_board = deepcopy(board)
        temp_board = makeMove(temp_board, move, player)[0]
        board_score = minimizeBeta(temp_board, depth - 1, alpha, beta, player, opponent)

        if board_score > best_score:
            best_score = board_score
            best_move = move

    return best_move


def minimizeBeta(board, depth, alpha, beta, player, opponent):
    valid_moves = getValidMoves(board)

    if depth == 0 or not valid_moves or gameIsOver(board):
        return utilityValue(board, player)

    for move in valid_moves:
        if alpha < beta:
            temp_board = deepcopy(board)
            temp_board = makeMove(temp_board, move, opponent)[0]
            board_score = maximizeAlpha(temp_board, depth - 1, alpha, beta, player, opponent)

            beta = min(beta, board_score)

    return beta


def maximizeAlpha(board, depth, alpha, beta, player, opponent):
    valid_moves = getValidMoves(board)

    if depth == 0 or not valid_moves or gameIsOver(board):
        return utilityValue(board, player)

    for move in valid_moves:
        if alpha < beta:
            temp_board = deepcopy(board)
            temp_board = makeMove(temp_board, move, player)[0]
            board_score = minimizeBeta(temp_board, depth - 1, alpha, beta, player, opponent)

            alpha = max(alpha, board_score)

    return alpha
