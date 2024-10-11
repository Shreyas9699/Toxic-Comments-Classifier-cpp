import numpy as np
import pandas as pd
import re

# Read train and test data
train_df = pd.read_csv('data/train.csv').fillna(' ')
test_df = pd.read_csv('data/test.csv').fillna(' ')

# Create DataFrames with 'comment_text' and 'toxic' columns
train_df = train_df[['comment_text', 'toxic']]
test_df = test_df[['comment_text']]

# Replace newline characters with spaces and convert to lowercase
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x)).str.lower()
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x)).str.lower()


# Remove special characters, excluding single quotes
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s']", '', x))
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s']", '', x))

# Write DataFrames to CSV files
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)