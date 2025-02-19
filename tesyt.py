import tensorflow as tf
import numpy as np

class BitBoard:
    def __init__(self):
        # Two bitboards: one for each player (Player 1 will be represented with 'x' and Player 2 with 'o')
        self.board1 = 0  # bits where Player 1 has played
        self.board2 = 0  # bits where Player 2 has played
        self.mask = 0    # overall occupancy (board1 OR board2)
        self.current_player = 1  # Player 1 starts
        self.moves = []  # history of moves (columns playe

# Load the trained model
model = tf.keras.models.load_model("connect4_model.h5")

# Generate a dummy 6x7 board (replace this with actual game input formatting)
dummy_board = np.random.randint(-1,2, size=(6, 7))  # Random board with values [0, 1, 2]
print(dummy_board)
# Flatten the board to match the model's expected shape (1, 42)
board_input = dummy_board.flatten().reshape(1, 42)

# Ensure the data type matches what the model expects (float32)
board_input = board_input.astype(np.float32)

# Make prediction
output = model.predict(board_input, verbose=0)

# Print output details
print("Model Output:", output)
print("Output Shape:", output.shape)

# Extract the first element if needed
if isinstance(output, np.ndarray) and output.shape[0] == 1:
    probabilities = output

    print("Extracted Probabilities:", probabilities)
