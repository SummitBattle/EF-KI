import os

import numpy as np


# Color definitions (for console output)
RED     = '\033[1;31;40m'
RED_BG  = '\033[0;31;47m'
BLUE_BG = '\033[0;34;47m'
YELLOW  = '\033[1;33;40m'
BLUE    = '\033[1;34;40m'
MAGENTA = '\033[1;35;40m'
CYAN    = '\033[1;36;40m'
WHITE   = '\033[1;37;40m'

# Board dimensions
BOARD_WIDTH  = 7
BOARD_HEIGHT = 6
# For bitboard representation we use 7 bits per column (an extra bit is used as a sentinel)
COL_SIZE = BOARD_HEIGHT + 1  # 7

# Precompute masks for each column:
# - bottom_mask: the bit corresponding to the bottom cell of the column.
# - column_mask: a mask that covers the 6 playable cells in the column.
# - top_mask: the bit corresponding to the top playable cell (if this is set, the column is full).
bottom_mask = [1 << (col * COL_SIZE) for col in range(BOARD_WIDTH)]
column_mask = [(((1 << BOARD_HEIGHT) - 1) << (col * COL_SIZE))
               for col in range(BOARD_WIDTH)]
top_mask = [1 << (col * COL_SIZE + BOARD_HEIGHT - 1) for col in range(BOARD_WIDTH)]

# FULL_MASK is the union of all playable cells.
FULL_MASK = 0
for col in range(BOARD_WIDTH):
    FULL_MASK |= column_mask[col]

def is_winning_position(pos):
    """
    Efficiently checks if 'pos' (a bitboard) contains a 4-in-a-row.
    Uses bitwise operations to detect vertical, horizontal, and diagonal wins.
    """
    # Vertical (shift by 1)
    if pos & (pos >> 1) & (pos >> 2) & (pos >> 3):
        return True
    # Horizontal (shift by COL_SIZE, i.e., 7)
    if pos & (pos >> COL_SIZE) & (pos >> (2 * COL_SIZE)) & (pos >> (3 * COL_SIZE)):
        return True
    # Diagonal (\) (shift by COL_SIZE+1, i.e., 8)
    if pos & (pos >> (COL_SIZE + 1)) & (pos >> (2 * (COL_SIZE + 1))) & (pos >> (3 * (COL_SIZE + 1))):
        return True
    # Diagonal (/) (shift by COL_SIZE-1, i.e., 6)
    if pos & (pos >> (COL_SIZE - 1)) & (pos >> (2 * (COL_SIZE - 1))) & (pos >> (3 * (COL_SIZE - 1))):
        return True
    return False


class BitBoard:
    def __init__(self):
        # Two bitboards: one for each player (Player 1 will be represented with 'x' and Player 2 with 'o')
        self.board1 = 0  # bits where Player 1 has played
        self.board2 = 0  # bits where Player 2 has played
        self.mask = 0    # overall occupancy (board1 OR board2)
        self.current_player = 1  # Player 1 starts
        self.moves = []  # history of moves (columns played)

    def can_play(self, col):
        """Return True if there is room in the given column (0-indexed)."""
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
            raise ValueError(f"Column {col+1} is full")
        # Determine the move bit: the lowest empty cell in column 'col'
        move = (self.mask + bottom_mask[col]) & column_mask[col]
        if self.current_player == 1:
            self.board1 |= move
        else:
            self.board2 |= move
        self.mask |= move
        self.moves.append(col)
        # Determine the row where the move was played.
        # (Each column occupies bits col*COL_SIZE to col*COL_SIZE+5;
        #  the row is the offset within the column.)
        row = (move.bit_length() - 1) - (col * COL_SIZE)
        # Check if the player who just played wins
        if self.current_player == 1:
            win = is_winning_position(self.board1)
        else:
            win = is_winning_position(self.board2)
        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return row, col, win

    def is_board_filled(self):
        """Return True if the board is completely filled."""
        return self.mask == FULL_MASK

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



    def get_board_array(self):
        """
        Convert the bitboard into a 2D numpy array representation.
        We'll use:
          - 0 for an empty cell,
          - 1 for a cell occupied by Player 1,
          - 2 for a cell occupied by Player 2.
        Note: The board is stored with row 0 at the bottom.
        """
        board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        for col in range(BOARD_WIDTH):
            for r in range(BOARD_HEIGHT):
                bit = 1 << (col * COL_SIZE + r)
                if self.board1 & bit:
                    board[r, col] = 1
                elif self.board2 & bit:
                    board[r, col] = 2
        return board


    def simulate_move(self, col):
        """Returns a new BitBoard with the move applied (without modifying original)."""
        new_board = BitBoard()
        new_board.board1 = self.board1
        new_board.board2 = self.board2
        new_board.mask = self.mask
        new_board.current_player = self.current_player

        # Apply move using bitwise operations
        move = (self.mask + bottom_mask[col]) & column_mask[col]
        if self.current_player == 1:
            new_board.board1 |= move
        else:
            new_board.board2 |= move
        new_board.mask |= move
        new_board.current_player = 3 - self.current_player  # Switch player

        return new_board

