import os
import random

import numpy as np

from utility_functions import countSequence

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
    zobrist_table = {}  # Store random values for hashing

    @classmethod
    def initialize_zobrist(cls):
        """Initialize Zobrist Hashing Table."""
        cls.zobrist_table = {}
        for col in range(BOARD_WIDTH):
            for row in range(BOARD_HEIGHT):
                for player in (1, 2):
                    cls.zobrist_table[(col, row, player)] = random.getrandbits(64)  # 64-bit random number

    def __init__(self):
        self.board1 = 0
        self.board2 = 0
        self.mask = 0
        self.current_player = 1
        self.moves = []
        self.lowest_empty_row = [BOARD_HEIGHT - 1] * BOARD_WIDTH
        self.hash_value = 0  # Store the hash of the board


    def can_play(self, col):
            """Check if there is room in the given column (0-indexed)."""
            return (self.mask & top_mask[col]) == 0

    def hash(self):
        """Returns the unique hash of the current board state."""
        return self.hash_value

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
            print(f"cannot play{col}")
            # Find the closest playable column to the center (column 3 in a 7-wide board)
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")

            # Sort by proximity to the center column (3)
            col = min(valid_moves, key=lambda c: abs(c - 3))
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

    def find_blocking_double_threat_move(self):
        """
        For each valid column, simulate the opponent playing that column *and then* making a second move.
        If by playing in that column the opponent would have at least 2 immediate winning moves on the next turn,
        then return that column index as the move to block the double threat.

        Returns:
            A column index (0-indexed) to block the double threat, or None if no such threat exists.
        """
        opponent = 2 if self.current_player == 1 else 1
        # Iterate through every valid move (column) that can be played.
        for col in self.get_valid_moves():
            # Copy board and simulate opponent playing in col.
            board_sim = self.copy()
            board_sim.current_player = opponent  # Force opponent to move.
            try:
                _, _, win_first = board_sim.play_move(col)
            except ValueError:
                continue
            # If opponent wins immediately by playing col, that’s an outright win—not a double threat.
            if win_first:
                continue

            # Now simulate giving the opponent a second move (i.e. skipping our turn)
            # Force the board to remain with the opponent.
            board_sim.current_player = opponent
            win_count = 0
            for opp_col in board_sim.get_valid_moves():
                test_board = board_sim.copy()
                test_board.current_player = opponent
                try:
                    _, _, win_second = test_board.play_move(opp_col)
                except ValueError:
                    continue
                if win_second:
                    win_count += 1

            # If by playing in column col the opponent would have at least 2 winning moves,
            # then playing col ourselves would block that potential double threat.
            if win_count >= 2:
                return col

        return None

    def find_double_threat_move(self):
        """
        For each valid move available to the AI, simulate playing that move and then,
        by skipping the opponent’s turn (i.e. forcing the turn back to the AI), count how many
        immediate winning moves the AI would have. If any move yields at least 2 winning moves,
        return that move (column index) as the one that creates a double threat.

        Returns:
            A column index (0-indexed) that creates a double threat for the AI, or None if no such move exists.
        """
        # Save the AI's identity.
        ai_player = self.current_player

        for col in self.get_valid_moves():
            # Copy the board and simulate playing the candidate move.
            board_after_move = self.copy()
            # The play_move call switches the turn after playing.
            _, _, immediate_win = board_after_move.play_move(col)

            # If the move itself is an immediate win, that’s even better.
            if immediate_win:
                return col

            # For simulation, force the turn back to the AI (i.e. skip the opponent).
            board_after_move.current_player = ai_player

            win_count = 0
            # Check each valid move from this new state.
            for next_col in board_after_move.get_valid_moves():
                test_board = board_after_move.copy()
                test_board.current_player = ai_player  # Force the AI to play again.
                try:
                    _, _, win_next = test_board.play_move(next_col)
                except ValueError:
                    continue
                if win_next:
                    win_count += 1

            # If at least 2 moves yield an immediate win, we have created a double threat.
            if win_count >= 2:
                return col

        return None
