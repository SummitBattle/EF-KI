from copy import deepcopy

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
COL_SIZE = BOARD_HEIGHT + 1  # = 7 (6 playable rows plus 1 extra bit per column)
AI_PLAYER = 'o'
HUMAN_PLAYER = 'x'


def in_bounds(r, c):
    """Return True if row r and column c are within the playable board."""
    return 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH


def sequenceExists(pos, r, c, dr, dc, length):
    """
    Check if starting at (r, c) and proceeding in direction (dr, dc)
    there are 'length' consecutive bits set in the bitboard 'pos'.
    """
    for i in range(length):
        rr = r + i * dr
        cc = c + i * dc
        if not in_bounds(rr, cc):
            return False
        # Compute the bit position: each column occupies COL_SIZE bits.
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

    Note that sequences may overlap. This is similar to your original approach.
    """
    # Choose the correct bitboard: here we assume that if player == HUMAN_PLAYER,
    # their pieces are stored in board1; otherwise in board2.
    pos = bitboard_obj.board1 if player == HUMAN_PLAYER else bitboard_obj.board2
    totalCount = 0
    # Loop over all playable cells (rows 0 to BOARD_HEIGHT-1, cols 0 to BOARD_WIDTH-1)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            # Compute the bit for cell (r, c)
            bit = 1 << (c * COL_SIZE + r)
            if pos & bit:
                if sequenceExists(pos, r, c, 1, 0, length):  # vertical
                    totalCount += 1
                if sequenceExists(pos, r, c, 0, 1, length):  # horizontal
                    totalCount += 1
                if sequenceExists(pos, r, c, 1, 1, length):  # positive diagonal
                    totalCount += 1
                if sequenceExists(pos, r, c, -1, 1, length):  # negative diagonal
                    totalCount += 1
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


def utilityValue(board, player):
    """ A utility fucntion to evaluate the state of the board and report it to the calling function,
        utility value is defined as the  score of the player who calles the function - score of opponent player,
        The score of any player is the sum of each sequence found for this player scalled by large factor for
        sequences with higher lengths.
    """
    if player == HUMAN_PLAYER: opponent = AI_PLAYER
    else: opponent = HUMAN_PLAYER

    playerfours    = countSequence(board, player, 4)
    playerthrees   = countSequence(board, player, 3)
    playertwos     = countSequence(board, player, 2)
    playerScore    = playerfours*99999 + playerthrees*999 + playertwos*99

    opponentfours  = countSequence(board, opponent, 4)
    opponentthrees = countSequence(board, opponent, 3)
    opponenttwos   = countSequence(board, opponent, 2)
    opponentScore  = opponentfours*99999 + opponentthrees*999 + opponenttwos*99

    if opponentfours > 0:
        #This means that the current player lost the game
        #So return the biggest negative value => -infinity
        return float('-inf')
    else:
        #Return the playerScore minus the opponentScore
        return playerScore - opponentScore

def gameIsOver(bitboard_obj):
    """
    Returns True if either player has at least one 4-in-a-row.
    """
    if countSequence(bitboard_obj, HUMAN_PLAYER, 4) >= 1:
        return True
    elif countSequence(bitboard_obj, AI_PLAYER, 4) >= 1:
        return True
    else:
        return False

def copy_board(bitboard):
    return deepcopy(bitboard)


def detect_threats(bitboard):
    """
    Identifies immediate win moves for AI and opponent.
    Returns:
        - ai_winning_moves: List of columns where AI can win
        - opponent_winning_moves: List of columns where opponent can win
    """
    ai_winning_moves = []
    opponent_winning_moves = []

    valid_moves = bitboard.get_valid_moves()

    for move in valid_moves:
        temp_board = copy_board(bitboard)

        # Check if AI wins
        temp_board.play_move(move)
        if gameIsOver(temp_board):
            ai_winning_moves.append(move)

        # Check if opponent wins
        temp_board = copy_board(bitboard)
        temp_board.play_move(move)
        if gameIsOver(temp_board):
            opponent_winning_moves.append(move)

    return ai_winning_moves, opponent_winning_moves
