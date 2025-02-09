BOARD_WIDTH = 7
BOARD_HEIGHT = 6
AI_PLAYER = 'o'
HUMAN_PLAYER = 'x'

def countSequence(board, player, length):
    """ Given the board state , the current player and the length of Sequence you want to count
        Return the count of Sequences that have the given length
    """
    def verticalSeq(row, col):
        """Return 1 if it found a vertical sequence with the required length """
        count = 0
        for rowIndex in range(row, BOARD_HEIGHT):
            if board[rowIndex][col] == board[row][col]:
                count += 1
            else:
                break
        if count >= length:
            return 1
        else:
            return 0

    def horizontalSeq(row, col):
        """Return 1 if it found a horizontal sequence with the required length """
        count = 0
        for colIndex in range(col, BOARD_WIDTH):
            if board[row][colIndex] == board[row][col]:
                count += 1
            else:
                break
        if count >= length:
            return 1
        else:
            return 0

    def negDiagonalSeq(row, col):
        """Return 1 if it found a negative diagonal sequence with the required length """
        count = 0
        colIndex = col
        for rowIndex in range(row, -1, -1):
            if colIndex > BOARD_WIDTH - 1:
                break
            elif board[rowIndex][colIndex] == board[row][col]:
                count += 1
            else:
                break
            colIndex += 1  # increment column when row is incremented
        if count >= length:
            return 1
        else:
            return 0

    def posDiagonalSeq(row, col):
        """Return 1 if it found a positive diagonal sequence with the required length """
        count = 0
        colIndex = col
        for rowIndex in range(row, BOARD_HEIGHT):
            if colIndex > BOARD_WIDTH - 1:
                break
            elif board[row][colIndex] == board[row][col]:
                count += 1
            else:
                break
            colIndex += 1  # increment column when row incremented
        if count >= length:
            return 1
        else:
            return 0

    totalCount = 0
    # for each piece in the board...
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            # ...that is of the player we're looking for...
            if board[row][col] == player:
                # check if a vertical streak starts at (row, col)
                totalCount += verticalSeq(row, col)
                # check if a horizontal four-in-a-row starts at (row, col)
                totalCount += horizontalSeq(row, col)
                # check if a diagonal (both +ve and -ve slopes) four-in-a-row starts at (row, col)
                totalCount += (posDiagonalSeq(row, col) + negDiagonalSeq(row, col))
    # return the sum of sequences of length 'length'
    return totalCount

def EndValue(board, player):
    """ A utility function to evaluate the state of the board and report it to the calling function,
        utility value is defined as the score of the player who calls the function - score of opponent player,
        The score of any player is the sum of each sequence found for this player scaled by a large factor for
        sequences with higher lengths.
    """
    if player == HUMAN_PLAYER:
        opponent = AI_PLAYER
    else:
        opponent = HUMAN_PLAYER

    if countSequence(board, player, 4) >= 1:
        return 1  # Win
    elif countSequence(board, opponent, 4) >= 1:
        return -2  # Loss
    else:
        return 0.5
    # Draw or intermediate state

def utilityValue(board, player):
    """ A utility function to evaluate the state of the board and report it to the calling function,
        utility value is defined as the score of the player who calls the function - score of opponent player,
        The score of any player is the sum of each sequence found for this player scaled by a large factor for
        sequences with higher lengths.
    """
    if player == HUMAN_PLAYER:
        opponent = AI_PLAYER
    else:
        opponent = HUMAN_PLAYER

    playerfours = countSequence(board, player, 4)
    playerthrees = countSequence(board, player, 3)
    playertwos = countSequence(board, player, 2)
    playerScore = playerfours * 100000 + playerthrees * 500 + playertwos * 20

    opponentfours = countSequence(board, opponent, 4)
    opponentthrees = countSequence(board, opponent, 3)
    opponenttwos = countSequence(board, opponent, 2)
    opponentScore = opponentfours * 100000 + opponentthrees * 500 + opponenttwos * 20

    if opponentfours > 0:
        # This means that the current player lost the game
        # So return the biggest negative value => -infinity
        return float('-inf')
    if playerfours > 0:
        return float('inf')
    else:
        # Return the playerScore minus the opponentScore
        return playerScore - opponentScore

def gameIsOver(board):
    """Check if there is a winner in the current state of the board"""
    if countSequence(board, HUMAN_PLAYER, 4) >= 1:
        return True
    elif countSequence(board, AI_PLAYER, 4) >= 1:
        return True
    else:
        return False