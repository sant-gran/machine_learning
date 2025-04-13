
import matplotlib.pyplot as plt
from preprocess import df_final  # Import full dataset
import seaborn as sns


# Load full dataset
df = df_final.copy()

# Group by user: mean and STD of their ratings

user_stats = df.groupby("user_id")["rating"].agg(["mean","std","count"]).reset_index()
user_stats.columns = ["user_id", "avg_rating", "std_rating", "num_ratings"]

print(" Sample user rating stats:")
print(user_stats.head())

# Histogram of user average rating behavior

plt.figure(figsize=(10,10))
sns.histplot(user_stats["avg_rating"], bins=25, kde=True)
plt.title("Distribution of Average Rating per User")
plt.ylabel("Average Ratings")
plt.xlabel("Number of Users")
plt.grid(True)
plt.show()

#Histogram of user Standard Dev

plt.figure(figsize=(10,10))
sns.histplot(user_stats["std_rating"].dropna(), bins=25, kde=True)
plt.title("Distribution of Rating Standard Deviation per User")
plt.ylabel("Standard Deviation of ratings")
plt.xlabel("Number of Users")
plt.grid(True)
plt.show()

# Check users who always rate the same score

flat_users = user_stats[user_stats["std_rating"]==0]
print(f" Users who always rate the same: {len(flat_users)}")
print(flat_users.head(10))

# Users who rated the most

top_users = user_stats.sort_values("num_rating", ascending=False).head(10)
print(" Top 10 most active users")
print(top_users)

# Visualize those top users' rating behavior
top_users_ids = top_users["user_id"].values
df_top = df[df["user_id"].isin(top_users_ids)]

plt.figure(figsize=(10,10))
sns.boxplot(data=df_top, x="user_id", y="rating")
plt.title("Rating patterns of top 5 users")
plt.xlabel("User ID")
plt.ylabel("Rating")
plt.show()

