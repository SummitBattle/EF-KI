import ConnectAlgorithm
import os

from minimaxAlphaBeta import MiniMaxAlphaBeta
from board import findFours, isColumnValid, printBoard, initializeBoard, isBoardFilled, makeMove
from utility_functions import AI_PLAYER, HUMAN_PLAYER

RED = '\033[1;31;40m'
YELLOW = '\033[1;33;40m'
BLUE = '\033[1;34;40m'
MAGENTA = '\033[1;36;40m'
WHITE = '\033[1;37;40m'

dir_path = os.getcwd()
os.chdir(dir_path)

global move_count
parent_node = None


def playerTurn(board):
    """Handles player's turn and returns updated board, win status, and move index."""
    while True:
        Col = input(YELLOW + 'Choose a Column between 1 and 7: ' + WHITE)

        if not Col.isdigit():
            print(MAGENTA + "Input must be an integer!" + WHITE)
            continue

        playerMove = int(Col) - 1

        if playerMove < 0 or playerMove > 6:
            print(MAGENTA + "Column must be between 1 and 7!" + WHITE)
            continue

        if not isColumnValid(board, playerMove):
            print(MAGENTA + "The Column you selected is full!" + WHITE)
            continue

        break

    board = makeMove(board, playerMove, HUMAN_PLAYER)[0]
    playerFourInRow = findFours(board)

    return board, playerFourInRow, playerMove


def playerWins(board):
    """Handles player win scenario."""
    printBoard(board)
    print('                    ' + BLUE + "HUMAN WINS !!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'

    if playagain:
        mainFunction()

    return 0


def aiTurn(board, move_count, playerMove, human_start):
    """Runs MCTS for AI's move and updates the board."""
    global parent_node

    depth = 5

    # Run MCTS to get AI's next move
    aiMove, parent_node = ConnectAlgorithm.start_MCTS(
        board, parent_node=parent_node, depth=depth, playerMove=playerMove, human_starts=human_start
    )

    # Make the move on the board
    board, _, _ = makeMove(board, aiMove, AI_PLAYER)

    # Check for AI's four-in-a-row situation
    aiFourInRow = findFours(board)

    return board, aiFourInRow, aiMove


def aiWins(board):
    """Handles AI win scenario."""
    printBoard(board)
    print('                     ' + RED + "AI WINS !!!!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'

    if playagain:
        mainFunction()

    return 0


def mainFunction():
    """Main game loop."""
    os.system('cls' if os.name == 'nt' else 'clear')

    board = initializeBoard()
    printBoard(board)
    move_count = 0
    parent_node = None
    playerMove = None  # Initialize the variable to avoid undefined references

    whileCondition = 1
    human_starts = input(YELLOW + 'DO YOU WANT TO START (y/n)? ' + WHITE).lower() == 'y'

    while whileCondition:
        if isBoardFilled(board):
            print("GAME OVER\n")
            break

        if human_starts:
            # Player Turn
            board, playerFourInRow, playerMove = playerTurn(board)
            move_count += 1

            if playerFourInRow:
                whileCondition = playerWins(board)
                if whileCondition == 0:
                    break

            # AI Turn
            board, aiFourInRow, aiMove = aiTurn(board, move_count, playerMove, human_starts)
            move_count += 1

            if aiFourInRow:
                whileCondition = aiWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)
            print(f" AI MOVE WAS: {aiMove + 1}")

        else:
            # AI Turn First
            board, aiFourInRow, aiMove = aiTurn(board, move_count, playerMove, human_starts)
            move_count += 1

            if aiFourInRow:
                whileCondition = aiWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)
            print(f" AI MOVE WAS: {aiMove + 1}")

            # Player Turn
            board, playerFourInRow, playerMove = playerTurn(board)
            move_count += 1

            if playerFourInRow:
                whileCondition = playerWins(board)
                if whileCondition == 0:
                    break

            printBoard(board)


# Start the game
mainFunction()
