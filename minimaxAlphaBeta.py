from board import *
from random import shuffle

def MiniMaxAlphaBeta(board,depth,player):

    valid_moves = getValidMoves(board)
    shuffle(valid_moves)
    best_move = valid_moves[0]
    best_score = float("-inf")

    alpha = float("-inf")
    beta = float ("inf")

    if player == AI_PLAYER: opponent = HUMAN_PLAYER
    else:
        opponent = AI_PLAYER

    for move in valid_moves:
        tempboard = makeMove(board,move,player)[0]
        board_score = minimizeBeta(tempboard, depth - 1, alpha, beta ,player, opponent)

        if board_score > best_score:
            best_score = board_score
            best_move = move

    return best_move


def minimizeBeta(tempboard, depth, a, b, player, opponent):

    valid_moves = []

    for col in range(7):
        if isValidMove(col,tempboard):
            move = makeMove(tempboard, col, player)[2]
            valid_moves.append(move)

    if depth == 0 or len(valid_moves) == 0 or gameIsOver(tempboard):
        return utilityValue(tempboard, player)

    beta = b
    valid_moves = getValidMoves(tempboard)

    for move in valid_moves:
        boardscore = float("inf")

        if a < beta:
            tempboard = makeMove(tempboard,move,opponent)[0]
            boardscore = maximizeAlpha(tempboard, depth - 1, a, beta, player, opponent)

        if boardscore < beta:
            beta = boardscore

    return beta

def maximizeAlpha(board, depth, a, b, player, opponent):
    validMoves = []
    for col in range(7):
        # if column col is a legal move...
        if isValidMove(col, board):
            # make the move in column col for curr_player
            temp = makeMove(board, col, player)[2]
            validMoves.append(temp)
    # check to see if game over
    if depth == 0 or len(validMoves) == 0 or gameIsOver(board):
        return utilityValue(board, player)

    alpha = a
    # if end of tree, evaluate scores
    for move in validMoves:
        boardScore = float("-inf")
        if alpha < b:
            tempBoard = makeMove(board, move, player)[0]
            boardScore = minimizeBeta(tempBoard, depth - 1, alpha, b, player, opponent)

        if boardScore > alpha:
            alpha = boardScore
    return alpha












