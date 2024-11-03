from datasets import load_dataset
import os
import zipfile
import pandas as pd

# Create directories if they don't exist
os.makedirs("artifacts/data_ingestion", exist_ok=True)

# Load the CNN/DailyMail dataset from Hugging Face (a common text summarization dataset)
print("Downloading dataset... This might take a few minutes...")
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Convert to DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Save as CSV files first
print("Saving training data...")
train_df.to_csv('artifacts/data_ingestion/train.csv', index=False)
print("Saving test data...")
test_df.to_csv('artifacts/data_ingestion/test.csv', index=False)

# Create a ZIP file containing both CSVs
print("Creating zip file...")
with zipfile.ZipFile('artifacts/data_ingestion/data.zip', 'w') as zipf:
    zipf.write('artifacts/data_ingestion/train.csv', 'train.csv')
    zipf.write('artifacts/data_ingestion/test.csv', 'test.csv')

print("Dataset downloaded and saved successfully!")