# Dimensions and player definitions
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
COL_SIZE = BOARD_HEIGHT + 1  # = 7 (6 playable rows plus 1 extra bit per column)
AI_PLAYER = 'o'
HUMAN_PLAYER = 'x'

from board import *


def in_bounds(r, c):
    """Return True if row r and column c are within the playable board."""
    return 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH


def sequence_exists(pos, r, c, dr, dc, length):
    for i in range(length):
        nr, nc = r + i * dr, c + i * dc
        if not (0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH):
            return False
        bit = 1 << (nc * COL_SIZE + nr)
        if not (pos & bit):
            return False
    return True

def countSequence(bitboard, player, length):
    """Count all sequences of exactly 'length' for the player."""
    pos = bitboard.board1 if player == 1 else bitboard.board2
    count = 0
    directions = [
        (1, 0),   # Vertical
        (0, 1),   # Horizontal
        (1, 1),   # Diagonal down-right
        (1, -1),  # Diagonal down-left
    ]

    for dr, dc in directions:
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if sequence_exists(pos, r, c, dr, dc, length):
                    count += 1
    return count
def EndValue(bitboard_obj, player):
    """
    Returns 1 if the given player has at least one 4-in-a-row,
    -1 if the opponent has a 4-in-a-row, and 0 otherwise.
    """
    opponent = AI_PLAYER if player == HUMAN_PLAYER else HUMAN_PLAYER

    if countSequence(bitboard_obj, player, 4) >= 1:
        return 1  # Win for player
    elif countSequence(bitboard_obj, opponent, 4) >= 1:
        return -1  # Loss for player
    else:
        return 0


def utilityValue(bitboard_obj, player):
    """
    Returns an evaluation of the current board state from the perspective of 'player'.

    The score is computed by counting sequences of length 2, 3, and 4
    and weighting them with increasing factors. (A 4-in-a-row returns +/-infinity.)
    """
    opponent = AI_PLAYER if player == HUMAN_PLAYER else HUMAN_PLAYER

    player_fours = countSequence(bitboard_obj, player, 4)
    player_threes = countSequence(bitboard_obj, player, 3)
    player_twos = countSequence(bitboard_obj, player, 2)
    playerScore = player_fours * 100000 + player_threes * 500 + player_twos * 20

    opponent_fours = countSequence(bitboard_obj, opponent, 4)
    opponent_threes = countSequence(bitboard_obj, opponent, 3)
    opponent_twos = countSequence(bitboard_obj, opponent, 2)
    opponentScore = opponent_fours * 100000 + opponent_threes * 500 + opponent_twos * 20

    if opponent_fours > 0:
        # The current player lost the game.
        return float('-inf')
    if player_fours > 0:
        # The current player won the game.
        return float('inf')
    return playerScore - opponentScore


def gameIsOver(bitboard_obj):
    """
    Returns True if either player has at least one 4-in-a-row.
    """
    if countSequence(bitboard_obj, HUMAN_PLAYER, 4) > 0 or countSequence(bitboard_obj, AI_PLAYER, 4) > 0:
        return True
    return False



def blockOrWinMove(bitboard_obj):
    ai = AI_PLAYER
    human = HUMAN_PLAYER

    # Check for immediate AI win
    for col in range(BOARD_WIDTH):
        if canPlay(bitboard_obj, col):
            temp_board = bitboard_obj.copy()
            temp_board.play_move(col)
            if countSequence(temp_board, ai, 4) > 0:
                return col  # Winning move

    # Check for human win needing block
    for col in range(BOARD_WIDTH):
        if canPlay(bitboard_obj, col):
            temp_board = bitboard_obj.copy()
            temp_board.play_move(col)
            # Check all human moves after AI's move
            for human_col in temp_board.get_valid_moves():
                human_temp = temp_board.copy()
                human_temp.play_move(human_col)
                if countSequence(human_temp, human, 4) > 0:
                    return col  # Block this column

    return -1  # No immediate win/block

def canPlay(bitboard_obj, col):
    """
    Checks if a move can be played in the given column.
    Returns True if the column is not full.
    """
    top_bit = 1 << ((col * COL_SIZE) + (BOARD_HEIGHT - 1))
    return not ((bitboard_obj.board1 | bitboard_obj.board2) & top_bit)
