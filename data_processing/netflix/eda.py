import pandas as pd
from preprocess import df_final  # Import full dataset

# Convert timestamp to datetime format
df_final["timestamp"] = pd.to_datetime(df_final["timestamp"], errors="coerce")

# Display dataset info
print("Dataset Info:")
print(df_final.info())

# Check for missing values
print("Missing values:")
print(df_final.isnull().sum())

# Summary Statistics
print("Summary Statistics:")
print(df_final.describe())


