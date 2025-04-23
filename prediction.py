'''
Objective: Train supervised learning algorithms and assess its performance.
Train a simple machine learning model and assess its performance.
Instructions
    1. Choose a classification target (e.g. classification of a value).
    2. Choose at least two simple model classes (e.g., decision tree, logistic regression).
    3. Split the dataset into training (e.g., 70%) and testing (e.g., 30%) sets.
    4. Train the model classes on the training set.
    5. Test the trained model on the the training set, testing set and the entire set.
'''
'''
SVM和xgboost优先
'''

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from data_loader import DataLoader
import numpy as np
import pandas as pd


dataloader=DataLoader()
train_raw_data=dataloader.load_data('raw/adult.data')
x, y = dataloader.preprocess_data_svm(train_raw_data)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

svm=SVC(
    kernel='linear', 
    C=1.0, 
    random_state=42)

svm.fit(train_x, train_y)
train_y_pred = svm.predict(train_x)
test_y_pred = svm.predict(test_x)

print("accuracy on train set: ", accuracy_score(train_y, train_y_pred))
print("accuracy on test set: ", accuracy_score(test_y, test_y_pred))
print("classification report on train set: ")
print(classification_report(train_y, train_y_pred))
