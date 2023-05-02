import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Read your data from the CSV file
data = pd.read_csv('output.csv')

# Separate the features (X) and the labels (y)
X = data.iloc[:, 1:-1].values  # Skip the first column (filename) and the last column (label)
y = data.iloc[:, -1].values  # Label column

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the stratified k-fold cross-validator
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Arrays to store performance metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Perform stratified k-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the SVM model
    model = SVC(kernel='linear')  # you can also try 'rbf', 'poly', etc.
    model.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

# Calculate the average performance metrics
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

print("Average accuracy: ", avg_accuracy)
print("Average precision: ", avg_precision)
print("Average recall: ", avg_recall)
print("Average F1 score: ", avg_f1_score)