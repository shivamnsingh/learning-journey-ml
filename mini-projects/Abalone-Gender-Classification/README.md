# Abalone Gender Classification: A Polars & SVM Approach

This project explores the **UCI Abalone Dataset** to predict the gender (Male, Female, or Infant) of abalones using their physical shell measurements. 

The goal is to determine if external physical attributes (length, weight, height) are sufficient to identify the biological sex of the specimen without invasive procedures.

## 🚀 The Tech Stack
* **Polars**: Used for high-performance data manipulation and feature engineering.
* **Scikit-Learn**: Implementation of the Support Vector Machine (SVM) classifier.
* **Seaborn/Matplotlib**: Data visualization and correlation analysis.

## 📊 Data Insights
Before modeling, I performed an exploratory data analysis (EDA) using Polars. Key findings included:
* **High Multicollinearity**: Measurements like `Length` and `Diameter` have a **0.98 correlation**, indicating redundant information.
* **Class Overlap**: While **Infants (I)** are physically distinct (smaller and lighter), **Males (M)** and **Females (F)** show significant overlap in all physical dimensions.



## 🧠 The Model: Support Vector Machine (SVM)
I utilized a **Support Vector Classifier (SVC)** with a **Radial Basis Function (RBF) kernel**. 

### Why SVM?
SVM is effective in high-dimensional spaces. By using the RBF kernel, the model projects the data into a higher dimension to find a non-linear boundary between the overlapping gender classes.

### Key Implementation Steps:
1. **Scaling**: Used `StandardScaler` to normalize weights (grams) and lengths (mm), as SVM is distance-sensitive.
2. **Pipelines**: Encapsulated scaling and modeling to prevent data leakage.
3. **One-Hot Encoding**: Leveraged Polars' `.to_dummies()` for categorical analysis.

## 📈 Performance Results
The model achieves an overall accuracy of **~57%**. 

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Infant (I)** | 0.70 | 0.85 | 0.77 |
| **Female (F)** | 0.47 | 0.66 | 0.55 |
| **Male (M)** | 0.53 | 0.26 | 0.35 |

**Conclusion**: The model is highly effective at identifying Infants but struggles to distinguish between adult Males and Females due to their biological similarity in shell structure.



## 🛠️ How to Run
1. Ensure you have the `abalone.csv` file in the root directory.
2. Install dependencies:
   ```bash
   pip install polars scikit-learn seaborn matplotlib
