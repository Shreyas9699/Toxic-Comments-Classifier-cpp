import numpy as np
import pandas as pd
import re


train_df = pd.read_csv('data/train.csv').fillna(' ')
x_df = train_df[['comment_text']]  # Create a DataFrame with 'comment_text' column
y_df = train_df[['toxic']]        # Create a DataFrame with 'toxic' column

test_df = pd.read_csv('data/test.csv')
x_test_df = test_df[['comment_text']]

# print(x_df.sample(10, random_state = 0))

x_df['comment_text'] = x_df['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x))
x_test_df['comment_text'] = x_test_df['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x))

# x_df['comment_text'] = x_df['comment_text'].str.replace(r'\n', '')
# x_test_df['comment_text'] = x_test_df['comment_text'].str.replace(r'\n', '')

x_df.loc[:, 'comment_text'] = (x_df['comment_text'].str.replace(r'\n', ' ').str.lower())
x_test_df.loc[:, 'comment_text'] = (x_test_df['comment_text'].str.replace(r'\n', ' ').str.lower())

# Write DataFrames to CSV files
x_df.to_csv('data/x_train.csv', index=False, header=None)
y_df.to_csv('data/y_train.csv', index=False, header=None)
x_test_df.to_csv('data/x_test.csv', index=False, header=None)