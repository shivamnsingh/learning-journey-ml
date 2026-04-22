# My Learnings

## 1. Data Preprocessing
- Differentiate the dataset into categorical columns and numerical columns.
- Always check if the data is imbalanced or not. If yes, use methods like SMOTE, downsampling, etc.

## 2. Decision Tree
- Post pruning is done on smaller datasets.
- Slight noise is fine but massive data loss is not.
- Ensemble technique works good in imbalance dataset

## 3. Traning Testing
- If applying tranformation in traning data use fit_transform
- If applying Transformation in testing data use only transform
