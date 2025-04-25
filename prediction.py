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

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from data_loader import DataLoader
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from data_loader import DataLoader
import numpy as np
import pandas as pd

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, random_state=42):
        """
        初始化 SVM 分类器
        """
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        self.dataloader = DataLoader()

    def load_and_preprocess_data(self, file_path, test_size=0.3):
        """
        加载并预处理数据
        """
        raw_data = self.dataloader.load_data(file_path)
        x, y = self.dataloader.preprocess_data(raw_data)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            x, y, test_size=test_size, random_state=self.random_state
        )

    def train(self):
        """
        训练 SVM 模型
        """
        self.model.fit(self.train_x, self.train_y)

    def evaluate(self):
        """
        评估模型性能
        """
        train_y_pred = self.model.predict(self.train_x)
        test_y_pred = self.model.predict(self.test_x)

        print("accuracy on train set: ", accuracy_score(self.train_y, train_y_pred))
        print("accuracy on test set: ", accuracy_score(self.test_y, test_y_pred))
        print("classification report on train set: ")
        print(classification_report(self.train_y, train_y_pred))

# # 使用示例
# if __name__ == "__main__":
#     classifier = SVMClassifier(kernel='linear', C=1.0, random_state=42)
#     classifier.load_and_preprocess_data('raw/adult.data')
#     classifier.train()
#     classifier.evaluate()

dataloader= DataLoader()
file_path = 'raw/adult.data'
raw_data = dataloader.load_data(file_path)
x, y = dataloader.preprocess_data(raw_data)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
dtrain,dtest = xgb.DMatrix(train_x, label=train_y), xgb.DMatrix(test_x, label=test_y)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

watchlist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 100
bst = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=10,verbose_eval=True)

train_y_pred = bst.predict(dtrain)
test_y_pred = bst.predict(dtest)

#evaluate
acc=accuracy_score(test_y, np.round(test_y_pred))
print("accuracy on test set: ", acc)
