"""
This module contains functions to prepare the data for training and testing the model. It includes functions to load the data, preprocess it, and split it into training and testing sets.
Step 1: Load the data from the CSV file.
Step 2: the data contains 2 columns:landmark_id,images. The landmark_id column contains the id of the landmark and the images column contains the image filenames separated by space.
Step 3: Preprocess the data by creating a new dataframe with two columns: 'landmark_id' and 'image'. The 'landmark_id' column will contain the id of the landmark and the 'image' column will contain the image filename.
Step 4: Split the data into training and testing sets using an 80-20 split. The training set will be used to train the model and the testing set will be used to evaluate the model's performance.

"""

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the data from the CSV file
file_path = '../data/gldv2_metadata/train_clean.csv'  # Update this path to your CSV file
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Create a list of rows, then convert to DataFrame (avoid DataFrame.append for pandas 2.x compatibility)
rows = []

# Step 3: Populate the preprocessed data
for _, row in data.iterrows():
    landmark_id = row['landmark_id']
    images = row['images'].split()
    for image in images:
        rows.append({'landmark_id': landmark_id, 'image': image})

preprocessed_data = pd.DataFrame(rows)

## the landmark_id is a string number, it is a categorical variable, we need to convert it to a numerical variable using label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
preprocessed_data['landmark_id_encoded'] = label_encoder.fit_transform(preprocessed_data['landmark_id'])

# Step 4: Split the data into training and testing sets
train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)