from minimaxAlphaBeta import *
import os

RED     = '\033[1;31;40m'
YELLOW  = '\033[1;33;40m'
BLUE    = '\033[1;34;40m'
MAGENTA = '\033[1;3yY6;40m'
WHITE   = '\033[1;37;40m'

dir_path = os.getcwd()
os.chdir(dir_path)

def playerTurn(board):
    Col = input(YELLOW + 'Choose a Column between 1 and 7: ' + WHITE)

    if not Col.isdigit():
        print(MAGENTA + "Input ynymust be an integer!" + WHITE)
        return playerTurn(board)

    playerMove = int(Col) - 1

    if playerMove < 0 or playerMove > 6:
        print(MAGENTA + "Column must be between 1 and 7!" + WHITE)
        return playerTurn(board)

    if not isColumnValid(board, playerMove):
        print(MAGENTA + "The Column you selected is full!" + WHITE)
        return playerTurn(board)

    board = makeMove(board, playerMove, HUMAN_PLAYER)[0]
    playerFourInRow = findFours(board)

    return board, playerFourInRow

def playerWins(board):
    printBoard(board)
    print('                    ' + BLUE + "HUMAN WINS !!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'

    if playagain:
        mainFunction()

    return 0

def aiTurn(board, depth, move):
    aiMove = MiniMaxAlphaBeta(board, depth, AI_PLAYER)
    board = makeMove(board, aiMove, AI_PLAYER)[0]
    aiFourInRow = findFours(board)

    return board, aiFourInRow

def aiWins(board):
    printBoard(board)
    print('                     ' + RED + "AI WINS !!!!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'

    if playagain:
        mainFunction()

    return 0

def getDepth():
    return 6  # Default depth for AI difficulty

def mainFunction():
    os.system('cls' if os.name == 'nt' else 'clear')

    board = initializeBoard()
    printBoard(board)
    depth = getDepth()
    move_count = 0  # Track the number of moves

    whileCondition = 1
    human_starts = input(YELLOW + 'DO YOU WANT TO START (y/n)? ' + WHITE).lower() == 'y'

    while whileCondition:
        if isBoardFilled(board):
            print("GAME OVER\n")
            break

        if human_starts:
            # Player Turn
            board, playerFourInRow = playerTurn(board)
            move_count += 1  # Increment move count


            if playerFourInRow:
                whileCondition = playerWins(board)
                if whileCondition == 0:
                    break

            # AI Turn
            board, aiFourInRow = aiTurn(board, depth, move_count)
            move_count += 1  # Increment move count

            if aiFourInRow:
                whileCondition = aiWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)

        else:
            # AI Turn First
            board, aiFourInRow = aiTurn(board, depth, move_count)
            move_count += 1  # Increment move count

            if aiFourInRow:
                whileCondition = aiWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)

            # Player Turn
            board, playerFourInRow = playerTurn(board)
            move_count += 1  # Increment move count

            if playerFourInRow:
                whileCondition = playerWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)

# Start the game
mainFunction()
