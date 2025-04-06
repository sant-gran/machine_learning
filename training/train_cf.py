"""In this script we will import the tensors from netflix/preprocess and the Neural Network from CFEncoder"""

# Since we are just experimenting with this I use sys and os (not ideal for production) this makes the project root importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import my modules (netflix/preprocess, CFEncoder)
from data_processing.netflix.preprocess import get_preprocessed_data
from hybrid_model.cf_encoder import CFEncoder


# Step 1: Load Preprocessed Data
Data = get_preprocessed_data()

train_user, train_movie, train_rating = Data["train"]
val_user, val_movie, val_rating = Data["val"]
num_users = Data["num_users"]
num_movies = Data["num_movies"]

# Create Dataloaders
train_dataset = TensorDataset(train_user, train_movie, train_rating)
val_dataset = TensorDataset(val_user, val_movie, val_rating)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle =True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize Model, Loss, Optimizer
model = CFEncoder(num_users=num_users, num_movies=num_movies)
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Store the losses values to later use them in model training monitoring
train_losses = []
val_losses = []

# Early Stopping Setup (Prevent overfitting, saves compute when the model stops improving)
best_val_loss = float('inf')
patience = 3
patience_counter = 0

# Save Path for Best Model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(project_root, "saved_models")
os.makedirs(model_dir, exist_ok=True) # Makes sure that there is a dir called "hybrid_model"

model_path = os.path.join(model_dir, "cf_encoder_model.pth")


# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for user, movie, rating in train_loader:
        output = model(user, movie)
        loss = criterion(output, rating.float()) # float conversion here for the ratings

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for user, movie, rating in val_loader:
            output = model(user, movie)
            loss = criterion(output, rating.float()) #float conversion here for the ratings
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f" Val Loss: {avg_val_loss:.4f}")

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
        print(" New best model saved! ")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f" Early stopping triggered at epoch {epoch+1}")
            break
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)


# Plot The Model's Behaviour
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()




