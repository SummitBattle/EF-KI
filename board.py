import numpy as np
import os
from copy import deepcopy

# Color definitions (for console output)
RED = '\033[1;31;40m'
RED_BG = '\033[0;31;47m'
BLUE_BG = '\033[0;34;47m'
YELLOW = '\033[1;33;40m'
BLUE = '\033[1;34;40m'
MAGENTA = '\033[1;35;40m'
CYAN = '\033[1;36;40m'
WHITE = '\033[1;37;40m'

# Board dimensions
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
# For bitboard representation we use 7 bits per column (an extra bit is used as a sentinel)
COL_SIZE = BOARD_HEIGHT + 1  # 7

# Precompute masks for each column:
bottom_mask = [1 << (col * COL_SIZE) for col in range(BOARD_WIDTH)]
column_mask = [(((1 << BOARD_HEIGHT) - 1) << (col * COL_SIZE)) for col in range(BOARD_WIDTH)]
top_mask = [1 << (col * COL_SIZE + BOARD_HEIGHT - 1) for col in range(BOARD_WIDTH)]
FULL_MASK = sum(column_mask)


class BitBoard:
    def __init__(self):
        # Two bitboards: one for each player (Player 1 will be represented with 'x' and Player 2 with 'o')
        self.board1 = 0  # bits where Player 1 has played
        self.board2 = 0  # bits where Player 2 has played
        self.mask = 0  # overall occupancy (board1 OR board2)
        self.current_player = 1  # Player 1 starts
        self.moves = []  # history of moves (columns played)
        self.lowest_empty_row = [BOARD_HEIGHT - 1] * BOARD_WIDTH  # Track the next available row per column

    def can_play(self, col):
        """Check if there is room in the given column (0-indexed)."""
        return (self.mask & top_mask[col]) == 0

    def get_valid_moves(self):
        """Return a list of valid columns (0-indexed) where a move can be made."""
        return [col for col in range(BOARD_WIDTH) if self.can_play(col)]

    def play_move(self, col):
        """
        Play a move in the given column.
          - Computes the bit corresponding to the lowest empty cell in that column.
          - Updates the current player's bitboard and the overall mask.
          - Returns the (row, col) where the piece was placed and a flag indicating a win.
        """
        if not self.can_play(col):
            raise ValueError(f"Column {col + 1} is full")

        # Get the row and bit corresponding to the move
        row = self.lowest_empty_row[col]
        move = (self.mask + bottom_mask[col]) & column_mask[col]

        # Update the board state
        if self.current_player == 1:
            self.board1 |= move
        else:
            self.board2 |= move
        self.mask |= move
        self.lowest_empty_row[col] -= 1  # Decrease the available row for this column

        self.moves.append(col)

        # Check for a win
        win = self.check_win()

        # Switch the current player
        self.current_player = 2 if self.current_player == 1 else 1

        return row, col, win

    def check_win(self):
        """Check if the current player has won."""
        if self.current_player == 1:
            return self.is_winning_position(self.board1)
        else:
            return self.is_winning_position(self.board2)

    def is_winning_position(self, pos):
        """Check whether the given bitboard 'pos' contains a winning 4-in-a-row."""
        # Vertical check
        m = pos & (pos >> 1)
        if m & (m >> 2):
            return True

        # Horizontal check
        m = pos & (pos >> COL_SIZE)
        if m & (m >> (2 * COL_SIZE)):
            return True

        # Diagonal check (/)
        m = pos & (pos >> (COL_SIZE - 1))
        if m & (m >> (2 * (COL_SIZE - 1))):
            return True

        # Diagonal check (\)
        m = pos & (pos >> (COL_SIZE + 1))
        if m & (m >> (2 * (COL_SIZE + 1))):
            return True

        return False

    def print_board(self):
        """Clear the screen and print the board in a human‑readable form."""
        os.system('cls' if os.name == 'nt' else 'clear')
        moves_played = len(self.moves)
        print('')
        print(YELLOW + '         ROUND #' + str(moves_played) + WHITE)
        print('')
        print("\t      1   2   3   4   5   6   7")
        print("\t      -   -   -   -   -   -   -")
        # Print rows from top (row 6 is the extra bit; playable rows are 5 down to 0)
        for r in range(BOARD_HEIGHT - 1, -1, -1):
            print(WHITE + "\t", r+1, ' ', end="")
            for col in range(BOARD_WIDTH):
                # Compute the bit corresponding to row r in column col
                bit = 1 << (col * COL_SIZE + r)
                if self.board1 & bit:
                    # Player 1 piece ('x') in blue
                    piece = BLUE + 'x' + WHITE
                elif self.board2 & bit:
                    # Player 2 piece ('o') in red
                    piece = RED + 'o' + WHITE
                else:
                    piece = ' '
                print("| " + piece, end=" ")
            print("|")
        print('')



    def copy(self):
        """
        Creates a fast, efficient copy of the BitBoard instance.
        """
        new_bitboard = BitBoard()
        new_bitboard.board1 = self.board1
        new_bitboard.board2 = self.board2
        new_bitboard.mask = self.mask
        new_bitboard.current_player = self.current_player
        new_bitboard.moves = self.moves[:]  # Shallow copy for moves
        new_bitboard.lowest_empty_row = self.lowest_empty_row[:]
        return new_bitboard

    def get_board_matrix(self):
        board_matrix = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                idx = r * BOARD_WIDTH + c
                if (self.board1 >> idx) & 1:
                    board_matrix[r, c] = 1
                elif (self.board2 >> idx) & 1:
                    board_matrix[r, c] = 2
        return board_matrix

    def is_board_filled(self):
        """Return True if the board is completely filled."""
        return self.mask == FULL_MASK

    def find_winning_or_blocking_move(bitboard):
        """
        Finds a move that either wins the game or blocks the opponent's win.
        Returns the column index (0-indexed) of the best move or None if no immediate threat is found.
        """
        for col in range(BOARD_WIDTH):
            if not bitboard.can_play(col):
                continue  # Skip full columns

            # Simulate move for the current player
            temp_board = bitboard.copy()
            _, _, win = temp_board.play_move(col)
            if win:
                return col  # Winning move found

            # Simulate move for the opponent to check if they have a winning move
            temp_board = bitboard.copy()
            temp_board.current_player = 2 if bitboard.current_player == 1 else 1  # Switch to opponent
            _, _, opponent_win = temp_board.play_move(col)
            if opponent_win:
                return col  # Blocking move found

        return None  # No immediate winning or blocking move