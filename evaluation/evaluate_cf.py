import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from hybrid_model.cf_encoder import CFEncoder
from data_processing.netflix.preprocess import get_preprocessed_data
from evaluation.metrics import rmse

# Load test tensors
Data = get_preprocessed_data()
test_user, test_movie, test_rating = Data["test"]

# Load Encoders
with open("raw_data/netflix/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)
with open("raw_data/netflix/movie_encoder.pkl", "rb") as f:
    movie_encoder = pickle.load(f)

# Dataset & Dataloader
test_dataset = TensorDataset(test_user, test_movie, test_rating)
test_loader = DataLoader(test_dataset, batch_size=64)

# Load Model
num_users = len(user_encoder.classes_)
num_movies = len(movie_encoder.classes_)

model = CFEncoder(num_users=num_users, num_movies=num_movies)
model_path = "saved_models/cf_encoder_pretest_backup.pth"
model.load_state_dict(torch.load(model_path))

# Device setup (M1, CUDA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

# Evaluate
all_preds = []
all_targets = []

with torch.no_grad():
    for user, movie, rating in test_loader:
        user, movie, rating = user.to(device), movie.to(device), rating.float().to(device)
        preds, _, _ = model(user, movie)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(rating.cpu().numpy())

rmse_value = rmse(all_preds, all_targets)
print(f"\nTest RMSE: {rmse_value:.4f}")

# Plot prediction accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(all_targets, all_preds, alpha=0.3)
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title(f"Predicted vs True Ratings (RMSE: {rmse_value:.4f})")
plt.grid(True)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.tight_layout()
plt.show()