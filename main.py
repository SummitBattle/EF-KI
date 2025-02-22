import math
import os
import ConnectAlgorithm
import minimaxAlphaBeta

from board import *  # import the bitboard implementation
import importlib
import board


# Global variable for MCTS
global parent_node
parent_node = None

RED     = '\033[1;31;40m'
RED_BG  = '\033[0;31;47m'
BLUE_BG = '\033[0;34;47m'
YELLOW  = '\033[1;33;40m'
BLUE    = '\033[1;34;40m'
MAGENTA = '\033[1;35;40m'
CYAN    = '\033[1;36;40m'
WHITE   = '\033[1;37;40m'

def humanTurn(bitboard):
    """
    Handles the human player's turn using the bitboard representation.
    The human is prompted until a valid (non-full) columny entered.
    """
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

    # The BitBoard's play_move method uses its current_player and then toggles it.
    row, col, win = bitboard.play_move(col)
    return bitboard, win, col


def playerWins(game_state):
    """
    Handles the human win scenario.
    """
    game_state.print_board()
    print(BLUE + "HUMAN WINS !!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
    if playagain:
        mainFunction()
    exit()


import math
import random


def aiTurn(game_state, move_count, last_human_move, human_starts, first_move):
    """
    Handles the AI's turn using the bitboard representation.
    On the very first move the AI selects the middle column (3);
    otherwise, it uses the best available algorithm to choose a move.
    """
    global parent_node

    # Check for immediate win or block first
    threat = game_state.find_winning_or_blocking_move()
    if threat is not None:
        ai_move = threat
        parent_node = None
    elif first_move:
        ai_move = 3  # Choose the middle column on the first move
    elif move_count >= 20:
        ai_move = minimaxAlphaBeta.minimax_alpha_beta(game_state, 9, -math.inf, math.inf, game_state.current_player)[0]
    else:
        ai_move, _ = ConnectAlgorithm.start_MCTS(
            game_state,
            parent_node=parent_node,
            playerMove=last_human_move,
            human_starts=human_starts
        )

    try:

        row, col, win = game_state.play_move(ai_move)
    except:
        print("CANT PLAY MOVE, MOVE IS")
        print(ai_move)
        print("BOARD IS")
        game_state.print_board()
    return game_state, win, ai_move
def aiWins(game_state):
    """
    Handles the AI win scenario.
    """
    game_state.print_board()
    print(RED + "AI WINS !!!!\n" + WHITE)
    playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
    if playagain:
        mainFunction()
    exit()


def mainFunction():
    """
    Main game loop using the bitboard representation.
    The human and AI are assigned to different players depending on whether
    the human chooses to start.
      - If human_starts is True: human is Player 1 (bitboard.current_player == 1)
      - Otherwise: AI is Player 1 and human is Player 2.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    human_starts = input(YELLOW + 'DO YOU WANT TO START (y/n)? ' + WHITE).lower() == 'y'

    # Create a new bitboard instance.
    game_state = BitBoard()
    move_count = 0
    global parent_node
    parent_node = None
    last_human_move = None
    first_move = True

    game_state.print_board()

    while not game_state.is_board_filled():
        if human_starts:
            # When human starts, human is Player 1 and AI is Player 2.
            if game_state.current_player == 1:  # Human's turn
                game_state, win, human_move = humanTurn(game_state)
                move_count += 1
                last_human_move = human_move
                game_state.print_board()
                if win:
                    playerWins(game_state)
            else:  # AI's turn
                game_state, win, ai_move = aiTurn(game_state, move_count, last_human_move, human_starts, first_move)
                move_count += 1
                first_move = False
                game_state.print_board()
                print(f" AI MOVE WAS: {ai_move + 1}")
                if win:
                    aiWins(game_state)
        else:
            # When human does not start, AI is Player 1 and human is Player 2.
            if game_state.current_player == 1:  # AI's turn
                game_state, win, ai_move = aiTurn(game_state, move_count, last_human_move, human_starts, first_move)
                move_count += 1
                first_move = False
                game_state.print_board()
                print(f" AI MOVE WAS: {ai_move + 1}")
                if win:
                    aiWins(game_state)
            else:  # Human's turn
                game_state, win, human_move = humanTurn(game_state)
                move_count += 1
                last_human_move = human_move
                game_state.print_board()
                if win:
                    playerWins(game_state)
    else:
        print("GAME OVER\nIt's a draw!")
        playagain = input(YELLOW + 'DO YOU WANT TO PLAY AGAIN (y/n)? ' + WHITE).lower() == 'y'
        if playagain:
            mainFunction()
        else:
            exit()


# Start the game
if __name__ == "__main__":
    mainFunction()
