import json
import numpy as np

from utility_functions import gameIsOver


def initialize_board():
    """Creates an empty 6x7 Connect Four board filled with zeros."""
    return np.zeros((6, 7), dtype=int)

def self_play(initial_state, num_games=10, output_file="self_play_results.json"):
    results = []

    for game_id in range(num_games):
        print(f"Starting game {game_id + 1}...")
        game_state = copy_board(initial_state)
        root = Node(game_state, done=False, parent_node=None, action_index=None, starting_player=AI_PLAYER)
        move_sequence = []
        current_player = AI_PLAYER

        while not gameIsOver(game_state):
            root.explore(min_rollouts=500, max_time=2.0, batch_size=64)
            best_child, move = root.next()
            move_sequence.append((current_player, move))
            game_state.play_move(move)
            root = best_child
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        winner = EndValue(game_state, AI_PLAYER)
        results.append({
            "game_id": game_id + 1,
            "moves": move_sequence,
            "winner": "AI" if winner == 1 else "Human" if winner == 0 else "Draw"
        })

        print(f"Game {game_id + 1} finished. Winner: {'AI' if winner == 1 else 'Human' if winner == 0 else 'Draw'}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Self-play results stored in {output_file}")

# Example usage
# initial_state = initialize_board()
# self_play(initial_state, num_games=10)
