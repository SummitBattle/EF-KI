import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Load dataset
data_file = "connect4_data.csv"  # Update with your file path
data = pd.read_csv(data_file, header=None).values

# Hyperparameters
input_dim = 42  # Board size (flattened)
output_dim = 7  # Columns to drop a piece
batch_size = 64
learning_rate = 0.001
epochs = 20
save_every = 1000  # Save every N batches

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model
q_network = QNetwork(input_dim, output_dim).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Check if there is a saved checkpoint
checkpoint_path = "q_network.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    q_network.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}...")
else:
    start_epoch = 0

# Training loop (batch-wise training)
for epoch in range(start_epoch, epochs):
    np.random.shuffle(data)  # Shuffle data each epoch
    total_loss = 0
    for batch_idx in range(0, len(data), batch_size):
        batch = data[batch_idx : batch_idx + batch_size]

        # Prepare input (board state) and target (winning move)
        boards = torch.FloatTensor(batch[:, :42]).to(device)
        targets = torch.LongTensor(batch[:, 42]).to(device)

        # Forward pass
        outputs = q_network(boards)
        loss = loss_fn(outputs, torch.nn.functional.one_hot(targets, num_classes=7).float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Save model checkpoint every N batches
        if batch_idx % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": q_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at batch {batch_idx}")

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / (len(data) // batch_size)}")

# Final model save
torch.save(q_network.state_dict(), "final_q_network.pth")
print("Training complete! Model saved as final_q_network.pth")
