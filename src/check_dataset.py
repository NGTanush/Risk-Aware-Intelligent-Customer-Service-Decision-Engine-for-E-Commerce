import pandas as pd

# Location of our dataset
file_path = "data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Basic information
print("\n===== DATASET INFORMATION =====")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== INTENT DISTRIBUTION =====")
print(df["intent"].value_counts())

print("\n===== CATEGORY DISTRIBUTION =====")
print(df["category"].value_counts())