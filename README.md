# 🏥 Machine Learning Model for Predicting Cardiovascular Risk

## 📌 Project Overview
This project develops a **machine learning-based cardiovascular risk prediction model** that analyzes **lifestyle and personal health factors** to predict whether an individual has a **Low, Medium, or High risk** of developing cardiovascular disease (CVD). 

The goal is to provide a **faster, more accessible, and data-driven** alternative to traditional medical diagnosis methods.

---

## 📊 Key Features
✔ **Predictive Analysis**: Uses machine learning to classify cardiovascular risk levels.  
✔ **Data-Driven Insights**: Trained on **2,100 samples** with **17 lifestyle and health-related features**.  
✔ **Multiple ML Models**: Evaluates **Logistic Regression, k-Nearest Neighbors (KNN), and Random Forest** to identify the best-performing model.  
✔ **Feature Engineering**: Implements **data preprocessing, encoding, scaling, and feature selection** to improve model accuracy.  
✔ **Model Performance Evaluation**: Utilizes **confusion matrix, precision-recall curve, ROC-AUC, and cross-validation** to ensure robustness.  
✔ **Fine-Tuning & Optimization**: Uses **GridSearchCV and RandomizedSearchCV** to find the optimal hyperparameters.  



## 🔬 Methodology

### **1️⃣ Data Preprocessing**
- **Feature Selection**: Identifies key lifestyle and demographic features.
- **Data Cleaning**: Checks for missing values and handles categorical & numerical features.
- **Feature Scaling & Encoding**: Standardizes numerical data and applies one-hot encoding to categorical data.
- **Data Splitting**: Divides dataset into **training (80%)** and **testing (20%)**.

### **2️⃣ Machine Learning Models**
- **Logistic Regression**: Efficient for classification tasks, providing probability-based predictions.
- **Random Forest Classifier**: An ensemble learning method that improves accuracy and reduces overfitting.
- **k-Nearest Neighbors (KNN)**: Simple distance-based model for classification.

### **3️⃣ Model Evaluation**
- **Performance Metrics**: Uses **accuracy, precision, recall, F1-score, and AUC-ROC curves** to compare models.
- **Cross-Validation**: Ensures model generalization using **5-fold cross-validation**.
- **Hyperparameter Tuning**: Optimizes model parameters with **GridSearchCV and RandomizedSearchCV**.



## 📈 Results
- **Best Model**: Logistic Regression outperformed the other models with **high precision and recall** across all cardiovascular risk categories.
- **AUC Scores**: The Logistic Regression model achieved **AUC scores close to 1**, indicating strong classification ability.
- **Feature Importance**: The most influential predictor of cardiovascular risk was **weight (kg)**.



## 🛠️ Technologies Used
- **Programming Language**: Python  
- **Libraries**:
  - `scikit-learn`
  - `NumPy`
  - `pandas`
  - `Matplotlib`
  - `seaborn`
- **Machine Learning Algorithms**: Logistic Regression, Random Forest, KNN
- **Data Preprocessing**: `StandardScaler`, `OneHotEncoder`
- **Model Optimization**: `GridSearchCV`, `Cross-Validation`
- **Performance Evaluation**: Confusion Matrix, Precision-Recall Curve, ROC Curve

---

## 📄 Authors
- ***Leong Yee Chung***
- ***Ang Chin Siang***
- ***Lee Jia Jie***
