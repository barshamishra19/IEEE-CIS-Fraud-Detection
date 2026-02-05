# 💳 IEEE-CIS Fraud Detection

An advanced machine learning pipeline designed to detect fraudulent transactions in large-scale financial datasets. This project was developed for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) competition.

## 📊 Project Scope

Transaction data is highly imbalanced and complex. This solution implements a robust preprocessing and classification pipeline to accurately assign a probability of fraud to each transaction.

### Highlights:
- **Data Integration**: Merges massive transaction and identity tables (600k+ rows).
- **Intelligent Preprocessing**: Automated handling of missing values (50%+ threshold) and outliers via capping.
- **Feature Engineering**: Implements Log transformations for skewed financial data.
- **Dimensionality Reduction**: Uses **Principal Component Analysis (PCA)** to handle high-cardinality features and reduce noise.

## 🧠 Model Architecture

The core of the system is a **Logistic Regression** model optimized with the `saga` solver and `L2` regularization. The pipeline is built using Scikit-Learn's `Pipeline` and `ColumnTransformer` for maximum reproducibility.

### Pipeline Steps:
1. **Imputation**: Median for numerical data, Most Frequent for categorical.
2. **Scaling**: StandardScaler for all numerical features.
3. **Encoding**: OneHotEncoding for categorical variables.
4. **Compression**: PCA reduces the feature space to 50 principal components.
5. **Classification**: Logistic Regression predicts the `isFraud` probability.

## 🛠️ Setup & Requirements

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 🚀 Execution

The notebook follows a systematic "Day-by-Day" logic approach:
- **EDA**: Visualizing fraud distributions and transaction amounts.
- **Training**: Fits the model on labeled data.
- **Inference**: Processes test data (handling hyphen-to-underscore column naming fixes) and generates `submission.csv`.

## 📈 Performance
The model uses ROC-AUC as the primary metric (internal evaluation) and produces high-confidence probabilities suitable for Kaggle leaderboard submission.

---
*Built for the IEEE-CIS Fraud Detection Challenge.*
