# Credit Risk Prediction using Stacked Classifier and Filter-Based Feature Selection

This project implements a machine learning-based credit risk prediction system following the methodology from the paper  
**"A machine learning‑based credit risk prediction engine system using a stacked classifier and a filter‑based feature selection method"**,  
applying it to the UCI Australian Credit Approval dataset.

## 📌 Project Highlights

- 🧹 Data preprocessing: missing value imputation, label encoding, normalization  
- 🧪 Filter-based feature selection using ANOVA F-test  
- 🤖 Stacked ensemble model: Random Forest + Gradient Boosting (or XGBoost), with Logistic Regression as meta-classifier  
- 📈 Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- 💾 Model persistence: Save and load using Joblib  
- 🧠 Optional: Extend with SHAP/LIME, AutoML, and imbalanced data techniques (e.g., SMOTE, CTGAN)

## 🧱 Methodology

### 1. Data Preprocessing
- Use the `ucimlrepo` library to fetch the Australian dataset (`id=143`)
- Flatten the target variable
- Encode nominal features
- Normalize continuous features

### 2. Filter-Based Feature Selection
- Apply `SelectKBest` with `f_classif` (ANOVA F-test)  
- Retain top 10 features based on F-score

### 3. Train-Test Split
- Use 80/20 split with `train_test_split`

### 4. Stacked Model Construction
- Base models:  
  - Random Forest (100 estimators)  
  - Gradient Boosting or XGBoost (100 estimators)  
- Meta-model: Logistic Regression  
- Combine using `StackingClassifier` with 5-fold CV

### 5. Model Evaluation
- Use metrics like accuracy, precision, recall, F1-score  
- Visualize confusion matrix  
- Save model with `joblib`
