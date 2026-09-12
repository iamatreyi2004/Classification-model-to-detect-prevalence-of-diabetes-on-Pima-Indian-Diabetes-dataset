import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
#Extracting dataset
data=pd.read_csv("Pima_data.csv")
data
#Extracting features and response
X=data.iloc[:,0:8]
y=data.iloc[:,8]
#Computation of correlation matrix
X=pd.DataFrame(X)
# Calculate the Pearson correlation matrix
correlation_matrix =X.corr(method='pearson')
print(correlation_matrix)# No significant correlatiob among predictor variables
# Standardize predictor
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)
# Train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(random_state=0)
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
print(y_pred,y_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=lm,X=X_train,y=y_train,cv=10) #10 folds cross validation
print("Accuracy {:2f} %".format(accuracies.mean()*100))
print("Standard Deviation {:2f}".format(accuracies.std()))

#Probit regression
model=sm.Probit(y_train, X_train)
result=model.fit()
print(result.summary())
y_prob = result.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))



#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 6: Train the model
rf.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = rf.predict(X_test)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
