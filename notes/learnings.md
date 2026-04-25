# My Learnings

## 1. Data Preprocessing
- Differentiate between categorical and numerical columns  
- Check for data imbalance; use SMOTE, downsampling, etc.

## 2. Decision Tree
- Post pruning is preferred for smaller datasets  
- Slight noise is acceptable; avoid major data loss  
- Ensemble techniques work well on imbalanced data  

## 3. Training & Testing
- Use `fit_transform` on training data  
- Use `transform` only on testing data  

## 4. Coding Practice
- Convert repeated code into functions for reusability, clean structure, and scalability

## 5. Concept I Keep Forgetting
- Bias is the difference between expected model prediction and true value
- Variance = how much your prediction “jumps” when data changes
- High Bias + Low Variance (Underfitting)
- Low Bias + High Variance (Overfitting)
- Low Bias + Low Variance (Ideal model)
