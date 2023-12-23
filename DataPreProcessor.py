import numpy as np
import pandas as pd
import re


train_df = pd.read_csv('data/train.csv').fillna(' ')
x_df = train_df[['comment_text']]  # Create a DataFrame with 'comment_text' column
y_df = train_df[['toxic']]        # Create a DataFrame with 'toxic' column

test_df = pd.read_csv('data/test.csv')
x_test_df = test_df[['comment_text']]
y_test_df = train_df[['toxic']]

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
y_test_df.to_csv('data/y_test.csv', index=False, header=None)


toxic_comments = train_df[train_df['toxic'] == 1][['comment_text']]
NON_toxic_comments = train_df[train_df['toxic'] == 0][['comment_text']]

toxic_comments['comment_text'] = toxic_comments['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x))
NON_toxic_comments['comment_text'] = NON_toxic_comments['comment_text'].apply(lambda x: re.sub(r'\n', r'\\n', x))

toxic_comments.loc[:, 'comment_text'] = (toxic_comments['comment_text'].str.replace(r'\n', ' ').str.lower())
NON_toxic_comments.loc[:, 'comment_text'] = (NON_toxic_comments['comment_text'].str.replace(r'\n', ' ').str.lower())

toxic_comments.to_csv('data/toxic_comments.csv', index=False, header=None)
NON_toxic_comments.to_csv('data/non_toxic_comments.csv', index=False, header=None)

