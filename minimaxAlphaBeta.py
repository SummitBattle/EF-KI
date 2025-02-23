import time
from utility_functions import utilityValue, AI_PLAYER, gameIsOver

# Transposition Table (Memoization)
transposition_table = {}

def minimax_alpha_beta(bitboard, depth, alpha, beta, maximizingPlayer):
    board_hash = bitboard.hash()  # Ensure your BitBoard has a fast hash function

    # Check if the board state is already evaluated
    if board_hash in transposition_table:
        return transposition_table[board_hash]

    valid_moves = bitboard.get_valid_moves()
    if depth == 0 or gameIsOver(bitboard) or not valid_moves:
        return None, utilityValue(bitboard, AI_PLAYER)

    best_move = None
    best_score = float('-inf') if maximizingPlayer else float('inf')

    # Move Ordering: Prioritize center moves first
    valid_moves.sort(key=lambda move: abs(move - 3))

    if maximizingPlayer:
        for move in valid_moves:
            temp_board = bitboard.copy()
            _, _, win = temp_board.play_move(move)

            if win:
                transposition_table[board_hash] = (move, float('inf'))
                return move, float('inf')

            _, board_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False)

            if board_score > best_score:
                best_score = board_score
                best_move = move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Alpha-beta pruning

    else:
        for move in valid_moves:
            temp_board = bitboard.copy()
            _, _, win = temp_board.play_move(move)

            if win:
                transposition_table[board_hash] = (move, float('-inf'))
                return move, float('-inf')

            _, board_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True)

            if board_score < best_score:
                best_score = board_score
                best_move = move

            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha-beta pruning

    # Store result in transposition table
    transposition_table[board_hash] = (best_move, best_score)
    return best_move, best_score


def best_move_with_time_limit(bitboard, time_limit=2.0):
    """
    Iterative Deepening: Search with increasing depth until time runs out.
    """
    start_time = time.time()
    best_move = None
    depth = 1

    while time.time() - start_time < time_limit:
        move, _ = minimax_alpha_beta(bitboard, depth, float('-inf'), float('inf'), True)
        if move is not None:
            best_move = move  # Store best move from the last completed depth
        depth += 1  # Increase depth for next iteration

    return best_move
