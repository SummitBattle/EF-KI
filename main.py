
import MCTS_MS
from board import *  # import the bitboard implementation
from utility_functions import countSequence
RED     = '\033[1;31;40m'
BLUE_BG = '\033[0;34;47m'
YELLOW  = '\033[1;33;40m'
WHITE   = '\033[1;37;40m'

# Global variable for MCTS
parent_node = None


def humanTurn(bitboard):
    """Handles the human player's turn."""
    while True:
        col = input(YELLOW + 'Choose a Column between 1 and 7: ' + WHITE)
        if not col.isdigit():
            print(MAGENTA + "Input must be an integer!" + WHITE)
            continue
        col = int(col) - 1
        if col < 0 or col >= 7:
            print(MAGENTA + "Column must be between 1 and 7!" + WHITE)
            continue
        if not bitboard.can_play(col):
            print(MAGENTA + "The Column you selected is full!" + WHITE)
            continue
        break

    row, col, win = bitboard.play_move(col)
    return bitboard, win, col


def playerWins(game_state):
    """Handles the human win scenario."""
    game_state.print_board()
    print(BLUE + "HUMAN WINS !!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
    if playagain:
        mainFunction()
    exit()

def get_immediate_move(game_state):
    """
    Returns a tuple (ai_move, found) if an immediate threat move is found,
    and None otherwise.
    """
    # Check for immediate win or block
    move = game_state.find_winning_or_blocking_move()
    if move is not None:
        return move, True

    # Check for enemy double threat
    move = game_state.find_blocking_double_threat_move()
    if move:
        return move, True

    return None, False


def aiTurn(game_state, move_count, last_human_move, human_starts, first_move):
    """Handles the AI's turn."""
    global parent_node
    print(countSequence(game_state,1,2))
    ai_move, reset_parent = get_immediate_move(game_state)

    if reset_parent:
        parent_node = None
    elif first_move:
        ai_move = 3  # Middle column on the first move.
    elif move_count >= 40:
        ai_move = minimaxAlphaBeta.MiniMaxAlphaBeta(game_state, 8, float("inf"), float("-inf"), game_state.current_player)[0]

    else:
        ai_move = MCTS_MS.get_ai_move(game_state)


    row, col, win = game_state.play_move(ai_move)
    return game_state, win, ai_move




def aiWins(game_state):
    """Handles the AI win scenario."""
    print(RED + "AI WINS !!!!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
    if playagain:
        mainFunction()
    exit()


def mainFunction():
    """Main game loop using the bitboard representation."""
    os.system('cls' if os.name == 'nt' else 'clear')

    # Create a new bitboard instance
    game_state = BitBoard()
    move_count = 0
    global parent_node
    parent_node = None
    last_human_move = None
    first_move = True
    human_starts = input(YELLOW + 'DO YOU WANT TO START (y/n)? ' + WHITE).lower() == 'y'
    if human_starts:
        game_state.current_player = 1
    else:
        game_state.current_player = 2

    while not game_state.is_board_filled():
        if game_state.current_player == 1:  # Human's turn
            game_state, win, human_move = humanTurn(game_state)
            move_count += 1
            last_human_move = human_move
            if win:
                playerWins(game_state)
        else: # AI's turn
            game_state, win, ai_move = aiTurn(game_state, move_count, last_human_move, human_starts, first_move)
            move_count += 1
            first_move = False
            print(f"AI MOVE WAS: {ai_move + 1}")
            game_state.print_board()
            if win:
                aiWins(game_state)

    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
    if playagain:
        mainFunction()
    else:
        exit()


# Start the game
if __name__ == "__main__":
    mainFunction()
