1. Data Preprocessing  
- Differentiate between categorical and numerical columns.  
- Always check for data imbalance; if present, use SMOTE, downsampling, etc.  

2. Decision Tree  
- Post pruning is preferred for smaller datasets.  
- Slight noise is acceptable, but avoid major data loss.  
- Ensemble techniques perform well on imbalanced data.  

3. Training & Testing  
- Use fit_transform on training data.  
- Use only transform on testing data.  

4. Coding Practice  
- Convert repeated code into functions to make it reusable, clean, and scalable. 
