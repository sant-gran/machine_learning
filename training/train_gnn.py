"""Here we are going to train the model using the wrapped architecture with the hybrid scorer"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hybrid_model.gnn_encoder import GraphSAGEWithSimilarity
import matplotlib.pyplot as plt

# Load preprocessed data
data = torch.load("/Users/santiagog/Desktop/Python/machine_learning/raw_data/facebook/facebook_graph_data.pt", weights_only=False)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
data = data.to(device)


# Initialize models
embedding_dim = 64
gnn = GraphSAGEWithSimilarity(
    in_channels=data.x.size(1),
    hidden_channels=64,
    out_channels=embedding_dim,
    dropout=0.5,
    scorer_hidden_dim=32
).to(device)

# Hyperparameters to optimize the model
optimizer = torch.optim.Adam(gnn.parameters(), lr=5e-4, weight_decay=1e-5)

# Learning rate scheduler using Reduce learning plateau, when it reaches it reduces the learning rate by 50%
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Loss function
def compute_loss(pos_scores, neg_scores):
    loss_pos = -F.logsigmoid(pos_scores).mean()
    loss_neg = -F.logsigmoid(-neg_scores).mean()
    return loss_pos + loss_neg

# Validation function
@torch.no_grad()
def validate():
    gnn.eval()
    val_pos_scores = gnn(data.x, data.train_pos_edge_index, data.val_pos_edge_index)
    val_neg_scores = gnn(data.x, data.train_pos_edge_index, data.val_neg_edge_index)
    return compute_loss(val_pos_scores, val_neg_scores).item()

# Prepare for training
best_val_loss = float('inf')
patience = 8
patience_counter = 0
epochs = 100
train_losses = []
val_losses = []

# Create and save model in directory
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/gnn_encoder.pt"

# Training loop with early stopping + LR scheduling
for epoch in range(1, epochs + 1):
    gnn.train()
    optimizer.zero_grad()

    pos_scores = gnn(data.x, data.train_pos_edge_index, data.train_pos_edge_index)
    neg_scores = gnn(data.x, data.train_pos_edge_index, data.train_neg_edge_index)
    loss = compute_loss(pos_scores, neg_scores)

    loss.backward()
    optimizer.step()

    val_loss = validate()
    scheduler.step(val_loss)
    print("Current learning rate:", scheduler.get_last_lr()[0])

    # Save losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d}, Train loss: {loss.item():4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'model-state_dict': gnn.state_dict(),
        }, "Best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training Session Complete")

# plot training/validation losses

plt.figure(figsize=(8,8))
plt.plot(train_losses, label='train Loss')
plt.plot(val_losses, label ='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GNN Training Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




