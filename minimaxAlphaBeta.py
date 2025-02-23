from utility_functions import AI_PLAYER, gameIsOver, utilityValue

def minimax_alpha_beta(bitboard, depth, maximizingPlayer, alpha=float('-inf'), beta=float('inf'), transposition_table=None):
    best_move = None
    best_score = float('-inf') if maximizingPlayer else float('inf')

    # Check if the game is over or if max depth is reached
    if depth == 0 or gameIsOver(bitboard):
        return None, utilityValue(bitboard, AI_PLAYER)

    valid_moves = bitboard.get_valid_moves()
    if not valid_moves:
        return None, 0  # No valid moves, return neutral score

    # Move Ordering: Prioritize center moves for better pruning
    valid_moves.sort(key=lambda move: abs(move - 3))

    for move in valid_moves:
        temp_board = bitboard.copy()
        if not temp_board.can_play(move):
            continue  # Skip invalid moves

        _, _, win = temp_board.play_move(move)

        # If this move results in an immediate win, return it
        if win:
            return move, float('inf') if maximizingPlayer else float('-inf')

        _, board_score = minimax_alpha_beta(temp_board, depth - 1, not maximizingPlayer, alpha, beta, transposition_table)

        if maximizingPlayer:
            if board_score > best_score:
                best_score = board_score
                best_move = move
            alpha = max(alpha, best_score)
        else:
            if board_score < best_score:
                best_score = board_score
                best_move = move
            beta = min(beta, best_score)

        if alpha >= beta:
            break  # Alpha-beta pruning

    return best_move, best_score
