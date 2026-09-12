# Classification Model to Detect Diabetes

## Objective
The objective of this project is to develop a classification model
to classify an individual as Diabetic (1) or Non-Diabetic (0)
using the Pima Indian Diabetes Dataset.

## Dataset
The dataset contains information about individuals such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The response variable is Outcome:
- 1 = Diabetic
- 0 = Non-Diabetic

## Methodology

### 1. Data Loading
The dataset is loaded using pandas.

### 2. Feature and Response Selection
The first 8 columns are used as predictor variables (X),
and Outcome is used as the response variable (y).

### 3. Standardization
The predictor variables are standardized using StandardScaler.

### 4. Train-Test Split
The data is divided into training and testing sets.

### 5. Classification Model
A classification model is fitted using the training data.
We have used Logistic regression,Probit Regression and Random Forest 

### 6. Model Evaluation
The model is evaluated using:
- Accuracy score
- Confusion Matrix
- Classification Report

To reduce variabilty associated with the models k-fold cross validation is performed.

### 7.Conclusion
Random Forest classifies the outcome of being diabetic or non-diabetic the best with accuracy score of 79.8%
