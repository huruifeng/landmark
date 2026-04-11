# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pathlib import Path

# %%
df = pd.read_csv('../data/gldv2_metadata/train.csv')
df

## intersect the df with the images in the directory
image_dir = '../data/images_north_america/images'
all_images = set(p.stem for p in Path(image_dir).rglob('*.jpg')) # get all images

print('Total images in directory:', len(all_images))
print('Total entries in df:', len(df))

# %%
## Create a small subset of the df, only keep 60% of the df
df.rename(columns={'id': 'image_id'}, inplace=True)
df = df[df['image_id'].isin(all_images)]

print(df.shape)

df_subset = df.sample(frac=0.6, random_state=42)

counts = df_subset['landmark_id'].value_counts()

head_ids = counts[counts >= 10].index
tail_ids = counts[counts < 10].index

df_head = df_subset[df_subset['landmark_id'].isin(head_ids)]
df_tail = df_subset[df_subset['landmark_id'].isin(tail_ids)]

print('df_head:', df_head.shape)
print('df_tail:', df_tail.shape)

# %%
saved_dir = '../data/gldv2_200k'
os.makedirs(saved_dir, exist_ok=True)

## save the subset
df_head.to_csv(f'{saved_dir}/Metadata_200k.csv', index=False   )

# %%
## copy the images to the new directory
import shutil
for image_id in tqdm(df_head['image_id'], desc='Copying images'):
    src_path = f'{image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg'
    dst_path = f'{saved_dir}/images/{image_id}.jpg'
    shutil.copy(src_path, dst_path)

## split the subset into train, val, test
train_head, temp_df = train_test_split(
    df_head,
    test_size=0.2,
    stratify=df_head['landmark_id'],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['landmark_id'],
    random_state=42
)
print('train:', train_head.shape)
print('val:', val_df.shape)
print('test:', test_df.shape)

train_head.to_csv(f'{saved_dir}/train.csv', index=False)
val_df.to_csv(f'{saved_dir}/val.csv', index=False)
test_df.to_csv(f'{saved_dir}/test.csv', index=False)
# %%
