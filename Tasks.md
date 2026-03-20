## Intruduction
Image retrieval is a fundamental problem in computer vision: given a query image, find similar images in a large database. 
This is especially important for query images containing landmarks, which accounts for a large portion of what people like to photograph.

In this project, given query images and, for each query, are expected to retrieve all database images containing the same landmarks (if any).
it contains a much larger number of classes (there are a total of 15K classes in this challenge), and the number of training examples per class may not be very large.

## Dataset
The dataset is the data/gldv2_micro. 
- train.csv is for training, it has two columns 
    - filename: the image filename in data/gldv2_micro/images folder
    - landmark_id: it is the class lable 
- val.csv is for validation

## Task and Aims
Given a query image, find similar images in a large database.

