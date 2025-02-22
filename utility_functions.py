# Dimensions and player definitions
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
COL_SIZE = BOARD_HEIGHT + 1  # = 7 (6 playable rows plus 1 extra bit per column)
AI_PLAYER = 'o'
HUMAN_PLAYER = 'x'

from board import BitBoard

def in_bounds(r, c):
    """Return True if row r and column c are within the playable board."""
    return 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH


def sequenceExists(pos, r, c, dr, dc, length):
    """
    Check if starting at (r, c) and proceeding in direction (dr, dc)
    there are 'length' consecutive bits set in the bitboard 'pos'.
    """
    end_r = r + (length - 1) * dr
    end_c = c + (length - 1) * dc
    if not in_bounds(end_r, end_c):  # Check if the sequence will go out of bounds
        return False

    # Check for each cell in the sequence
    for i in range(length):
        rr = r + i * dr
        cc = c + i * dc
        bit = 1 << (cc * COL_SIZE + rr)
        if not (pos & bit):
            return False

    return True


def countSequence(bitboard_obj, player, length):
    """
    Count the number of sequences of consecutive pieces (of the given 'length')
    for the given player in the bitboard.

    The function inspects each playable cell; if that cell is occupied by the player,
    it checks in all four directions:
       - Vertical (downward): dr = 1, dc = 0
       - Horizontal (to the right): dr = 0, dc = 1
       - Positive diagonal (down-right): dr = 1, dc = 1
       - Negative diagonal (up-right): dr = -1, dc = 1
    """
    pos = bitboard_obj.board1 if player == HUMAN_PLAYER else bitboard_obj.board2
    totalCount = 0

    # Use a set to avoid re-checking the same position in different directions
    checked_positions = set()

    # Loop over all playable cells (rows 0 to BOARD_HEIGHT-1, cols 0 to BOARD_WIDTH-1)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            if (r, c) in checked_positions:
                continue  # Skip already checked cells
            # Compute the bit for cell (r, c)
            bit = 1 << (c * COL_SIZE + r)
            if pos & bit:
                # Check all directions (vertical, horizontal, diagonals)
                directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for dr, dc in directions:
                    if sequenceExists(pos, r, c, dr, dc, length):
                        totalCount += 1
                        # Mark all positions in this sequence as checked
                        for i in range(length):
                            rr = r + i * dr
                            cc = c + i * dc
                            checked_positions.add((rr, cc))

    return totalCount


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
    """
    Returns the first column that either:
    1. Wins the game for the current player.
    2. Blocks the opponent from winning.

    If no such move exists, returns -1.
    """

    # Check if there's a winning move for either player or if a block is needed
    for col in range(BOARD_WIDTH):
        if canPlay(bitboard_obj, col):
            temp_board = bitboard_obj.copy()

            # Check if AI wins by playing here
            temp_board.play_move(col)
            if countSequence(temp_board, AI_PLAYER, 4) > 0:
                return col  # Play the winning move

            # Check if Human wins by playing here (so we need to block)
            temp_board.play_move(col)
            if countSequence(temp_board, HUMAN_PLAYER, 4) > 0:
                return col  # Block the opponent’s win

    return -1  # No immediate win or threat found


def canPlay(bitboard_obj, col):
    """
    Checks if a move can be played in the given column.
    Returns True if the column is not full.
    """
    top_bit = 1 << ((col * COL_SIZE) + (BOARD_HEIGHT - 1))
    return not ((bitboard_obj.board1 | bitboard_obj.board2) & top_bit)
