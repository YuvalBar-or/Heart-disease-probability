
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Display the column names
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Display the dataset size
print("\nDataset Size:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Visualize the distribution of each column
for column in df.columns:
    plt.figure(figsize=(10, 5))
    if df[column].dtype == 'object':  # Categorical Data
        df[column].value_counts().plot(kind='bar', color='skyblue')
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Distribution of {column}")
        plt.xticks(rotation=45)
    else:  # Numerical Data
        plt.hist(df[column].dropna(), bins=20, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {column}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
