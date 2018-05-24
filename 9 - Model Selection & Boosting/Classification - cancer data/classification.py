# Classification

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Gradient Boosting to the Training set 
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''data Attribute info:
1. Sample code number: id number
2. Clump Thickness: 1-10
3. Uniformity of Cell Size: 1-10
4. Uniformity of Cell Shape: 1-10
5. Marginal Adhesion: 1-10
6. Single Epithelial Cell Size: 1-10
7. Bare Nuclei: 1-10
8. Bland Chromatin: 1-10
9. Normal Nucleoli: 1-10
10. Mitoses: 1-10
11. Class: (2 for Benign, 4 for malignant)

'''
# conclusion - PREDICTION MODEL is VERY ACCURATE (based on the confusion matrix)
y_pred_label = []
for i in y_pred:
    if y_pred[i] == 2:
        y_pred_label.append(['Malignant'])
    else:
        y_pred_label.append(['Benign'])
