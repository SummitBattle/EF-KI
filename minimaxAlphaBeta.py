from random import shuffle

from utility_functions import gameIsOver, can_play, BOARD_WIDTH, AI_PLAYER


def getValidMoves(bitboard_obj):
    return [col for col in range(BOARD_WIDTH) if can_play(bitboard_obj, col)]


def MiniMaxAlphaBeta(bitboard_obj, depth, maximizingPlayer, alpha=float('-inf'), beta=float('inf')):
    validMoves = getValidMoves(bitboard_obj)
    shuffle(validMoves)
    bestMove = None
    bestScore = float('-inf') if maximizingPlayer else float('inf')

    if depth == 0 or gameIsOver(bitboard_obj):
        return None, utilityValue(bitboard_obj, AI_PLAYER)

    for move in validMoves:
        tempBoard = bitboard_obj.copy()
        tempBoard.play_move(move)
        _, boardScore = MiniMaxAlphaBeta(tempBoard, depth - 1, not maximizingPlayer, alpha, beta)

        if maximizingPlayer:
            if boardScore > bestScore:
                bestScore = boardScore
                bestMove = move
            alpha = max(alpha, bestScore)
        else:
            if boardScore < bestScore:
                bestScore = boardScore
                bestMove = move
            beta = min(beta, bestScore)

        if beta <= alpha:
            break  # Alpha-beta pruning

    return bestMove, bestScore
