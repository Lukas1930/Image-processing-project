import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle

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

# Load the model from the file
with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the loaded model for predictions
y_pred = loaded_model.predict(X)

# Evaluate the model on the testing set
y_pred = loaded_model.predict(X)
accuracies.append(accuracy_score(y, y_pred))
precisions.append(precision_score(y, y_pred, average='weighted'))
recalls.append(recall_score(y, y_pred, average='weighted'))
f1_scores.append(f1_score(y, y_pred, average='weighted'))

# Calculate the average performance metrics
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

print("Average accuracy: ", avg_accuracy)
print("Average precision: ", avg_precision)
print("Average recall: ", avg_recall)
print("Average F1 score: ", avg_f1_score)