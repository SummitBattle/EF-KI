from utility_functions import utilityValue, AI_PLAYER, gameIsOver

def minimax_alpha_beta(bitboard, depth, alpha, beta, maximizingPlayer):
    best_move = None
    best_score = float('-inf') if maximizingPlayer else float('inf')

    valid_moves = bitboard.get_valid_moves()
    if depth == 0 or gameIsOver(bitboard) or not valid_moves:
        return None, utilityValue(bitboard, AI_PLAYER)

    # Move Ordering: Prioritize center moves (column 3 for a 7-wide board)
    valid_moves.sort(key=lambda move: abs(move - 3))

    if maximizingPlayer:
        for move in valid_moves:
            temp_board = bitboard.copy()
            _, _, win = temp_board.play_move(move)

            if win:
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
                return move, float('-inf')

            _, board_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True)

            if board_score < best_score:
                best_score = board_score
                best_move = move

            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha-beta pruning

    return best_move, best_score
