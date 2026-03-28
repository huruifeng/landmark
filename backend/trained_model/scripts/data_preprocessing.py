# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv('../data/gldv2_395k/train_filtered.csv')
df


# %%
counts = df['landmark_id'].value_counts()
counts

head_ids = counts[counts >= 10].index
tail_ids = counts[counts < 10].index

df_head = df[df['landmark_id'].isin(head_ids)]
df_tail = df[df['landmark_id'].isin(tail_ids)]

print('df_head:', df_head.shape)
print('df_tail:', df_tail.shape)
print('head classes:', df_head['landmark_id'].nunique())
print('tail classes:', df_tail['landmark_id'].nunique())


# %%
df_head.rename(columns={'id': 'image_id'}, inplace=True)

# 80% train, 20% temp
train_head, temp_df = train_test_split(
    df_head,
    test_size=0.2,
    stratify=df_head['landmark_id'],
    random_state=42
)

# temp -> 10% val, 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['landmark_id'],
    random_state=42
)

# rare classes only go to train
train_df = pd.concat([train_head, df_tail], ignore_index=True)

temp_counts = temp_df['landmark_id'].value_counts()
print(temp_counts.min())
print(temp_counts[temp_counts < 2].head(20))

print('train:', train_df.shape)
print('val:', val_df.shape)
print('test:', test_df.shape)

print('train landmarks:', train_df['landmark_id'].nunique())
print('val landmarks:', val_df['landmark_id'].nunique())
print('test landmarks:', test_df['landmark_id'].nunique())


train_ids = set(train_df['image_id'])
val_ids = set(val_df['image_id'])
test_ids = set(test_df['image_id'])

print('train ∩ val:', len(train_ids & val_ids))
print('train ∩ test:', len(train_ids & test_ids))
print('val ∩ test:', len(val_ids & test_ids))

train_df.to_csv('../data/gldv2_395k/train.csv', index=False)
val_df.to_csv('../data/gldv2_395k/val.csv', index=False)
test_df.to_csv('../data/gldv2_395k/test.csv', index=False)

# %%
