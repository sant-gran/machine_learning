import pandas as pd
import torch
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Load Netflix Dataset
data_file= "/Users/santiagog/Desktop/netflix_dataset/combined_data_1.txt"

with open(data_file, "r") as file:
    for _ in range(10):
        print(file.readline().strip())

data_folder = "/Users/santiagog/Desktop/netflix_dataset"

# List of file (combined_data 1 to 4)
data_files = [f for f in os.listdir(data_folder) if f.startswith("combined_data")]

# Initialize list for storing DataFrames
df_list = []

# Read and process each file
for file in data_files:
    file_path = os.path.join(data_folder, file)
    with open(file_path, "r") as f:
        movie_id = None
        data = []

        for i, line in enumerate(f):
            if i >= 1000: #stop after 1000 lines
                break
            line = line.strip()
            if line.endswith(":"):  # Movie ID line (e.g., "1:")
                movie_id = int(line.replace(":", ""))
            else:  # User raw_data (user_id, rating, timestamp)
                user_id, rating, timestamp = line.split(",")
                data.append((int(user_id), movie_id, int(rating), timestamp))

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["user_id", "movie_id", "rating", "timestamp"])
        df_list.append(df)

# Combine all DataFrames
df_combined = pd.concat(df_list, ignore_index=True)

# Check raw_data types
print(df_combined.dtypes)

# Display first rows
print(df_combined.head())


# Path to movie titles
titles_path = "/Users/santiagog/Desktop/netflix_dataset/movie_titles.csv"


# Read movies titles
df_movies = pd.read_csv(
    titles_path, names=["movie_id", "year", "title"], encoding="latin1", delimiter=";", nrows=1000,
    dtype = {"movie_id": int, "year": int, "title": str}
)

# Check raw_data types
print(df_movies.dtypes)
# Check movie_id raw_data type in both tables to see if there is a raw_data type mismatch
print(df_combined["movie_id"].dtype)
print(df_movies["movie_id"].dtype)

# Display first few rows
print(df_movies.head())


df_final = df_combined.merge(df_movies, on="movie_id", how="left")
df_final = df_final.dropna(subset=["year", "title"])

# Show first few rows
print(df_final.head())


"""Pytorch Preprocessing"""
# Here we take the raw raw_data and preprocess

# Create encoders
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

# Apply encoders to create numerical indices
df_final["user_idx"] = user_encoder.fit_transform(df_final["user_id"])
df_final["movie_idx"] = user_encoder.fit_transform(df_final["movie_id"])

# Check the transformation
print("Sample Mapping:")
print("Original User ID:", df_final["user_id"].iloc[0], "--> User Index:", df_final['user_idx'].iloc[0])
print("Original Movie ID:", df_final["movie_id"].iloc[0], "--> Movie Index:", df_final['movie_idx'].iloc[0])

# Save encoders if needed for inverse mapping later for 1) making predictions after training 2) Deploy the model for real-time use
joblib.dump(user_encoder, "/Users/santiagog/Desktop/Python/machine_learning/raw_data/netflix/user_encoder.pkl")
joblib.dump(movie_encoder, "/Users/santiagog/Desktop/Python/machine_learning/raw_data/netflix/movie_encoder.pkl")


# Data Normalization needed for NNs better model performance and stability (type: min - max normalization)
    # Apply min-max normalization to ratings
min_rating = df_final["rating"].min()
max_rating = df_final["rating"].max()

# Min - Max Formula: N = rating - min/ max - min
df_final["normalized_rating"] = (df_final["rating"]- min_rating) / (max_rating -min_rating)

# Show a sample
print("Normalized ratings:")
print(df_final[["rating" , "normalized_rating"]].head())

"""Splitting the data"""

# Split off the Test set (20%)
train_val_df, test_df = train_test_split(df_final, test_size =0.2, random_state=42)

# Split remaining 80% into 70/10 --> train + validation
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)


print(f"Train Set: {len(train_df)} rows")
print(f"Validation Set: {len(val_df)} rows ")
print(f"Test Set: {len(test_df)} rows")



"""Convert split sets into Pytorch Tensors!!!!! 🔥🔥🔥 yeaahh buddyyy lightweight baby"""

# Convert train set to tensors
train_user_tensor = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
train_movie_tensor = torch.tensor(train_df["movie_idx"].values, dtype=torch.long)
train_rating_tensor = torch.tensor(train_df["normalized_rating"].values, dtype=torch.float)

# Convert validation set to tensors
val_user_tensor = torch.tensor(val_df["user_idx"].values, dtype=torch.long)
val_movie_tensor = torch.tensor(val_df["movie_idx"].values, dtype=torch.long)
val_rating_tensor = torch.tensor(val_df["normalized_rating"].values, dtype=torch.float)

# Convert test set to tensors
test_user_tensor = torch.tensor(test_df["user_idx"].values, dtype=torch.long)
test_movie_tensor = torch.tensor(test_df["movie_idx"].values, dtype=torch.long)
test_rating_tensor = torch.tensor(test_df["normalized_rating"].values, dtype=torch.float)



# Check a few samples
print("Tensor Samples:")
print("User:", train_user_tensor[:5])
print("Movie:", train_movie_tensor[:5])
print("Ratings:", train_rating_tensor[:5])

# Package everything for external use (training files)
preprocessed_data = {
    "train": (train_user_tensor, train_movie_tensor, train_rating_tensor),
    "val": (val_user_tensor, val_movie_tensor, val_rating_tensor),
    "test": (test_user_tensor, test_movie_tensor, test_rating_tensor),
    "num_users": df_final["user_idx"].nunique(),
    "num_movies": df_final["movie_idx"].nunique(),

}

# Function that calls the package when imported
def get_preprocessed_data():
    return preprocessed_data



