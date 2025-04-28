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

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from data_loader import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class Classifier:
    def __init__(self, model, random_state=42):
        self.dataloader = DataLoader()
        self.model = model
        self.random_state = random_state
        self.results = None

    def set_model(self, model):
        """
        设置模型
        """
        self.model = model

    def load_and_preprocess_data(self, file_path, test_size=0.3):
        """
        加载并预处理数据
        """
        raw_data = self.dataloader.load_data(file_path)
        self.x, self.y = self.dataloader.preprocess_data(raw_data)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.x, self.y, test_size=test_size, random_state=self.random_state
        )

    def train(self):
        """
        训练 SVM 模型
        """
        self.model.fit(self.train_x, self.train_y)
        
    def predict(self, x):
        """
        预测
        """
        return self.model.predict(x)

    def evaluate(self):
        """
        评估模型性能
        """
        self.train_y_pred = self.model.predict(self.train_x)
        self.test_y_pred = self.model.predict(self.test_x)
        self.total_y_pred = self.model.predict(self.x)
        

        

        # print("accuracy on train set: ", accuracy_score(
        #     self.train_y, self.train_y_pred))
        # print("accuracy on test set: ", accuracy_score(
        #     self.test_y, self.test_y_pred))
        # print("accuracy on total set: ",
        #       accuracy_score(self.y, self.total_y_pred))
        # print("classification report on train set: ")
        # print(classification_report(self.train_y, self.train_y_pred))
        # print("classification report on test set: ")
        # print(classification_report(self.test_y, self.test_y_pred))
        # print("classification report on total set: ")
        # print(classification_report(self.y, self.total_y_pred))
        # print("confusion matrix on train set: ")
        # print(confusion_matrix(self.train_y, self.train_y_pred))
        # print("confusion matrix on test set: ")
        # print(confusion_matrix(self.test_y, self.test_y_pred))
        # print("confusion matrix on total set: ")
        # print(confusion_matrix(self.y, self.total_y_pred))

    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.show()

    def plot_decision_boundary(self, features, labels, title):
        """
        绘制决策边界
        """
        # 使用 PCA 降维到 2D
        pca = PCA(n_components=2, random_state=self.random_state)
        X2 = pca.fit_transform(features)
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = self.model.predict(pca.inverse_transform(
            np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, edgecolor='k')
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()


# 初始化 XGBoost 分类器
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    eta=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 初始化随机森林分类器
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

# 初始化 SVM 分类器
svm_model = SVC(kernel='linear', random_state=42)


classifier = Classifier(svm_model, random_state=42)
classifier.load_and_preprocess_data('raw/adult.data')
classifier.train()
classifier.evaluate()
classifier.plot_confusion_matrix(
    classifier.train_y, classifier.train_y_pred, title='SVM Train Confusion Matrix')
classifier.plot_decision_boundary(
    classifier.train_x, classifier.train_y, title='SVM Decision Boundary')

classifier.set_model(rf_model)
classifier.train()
classifier.evaluate()
classifier.plot_confusion_matrix(
    classifier.train_y, classifier.train_y_pred, title='Random Forest Train Confusion Matrix')
classifier.plot_decision_boundary(
    classifier.train_x, classifier.train_y, title='Random Forest Decision Boundary')


classifier.set_model(xgb_model)
classifier.train()
classifier.evaluate()
classifier.plot_confusion_matrix(
    classifier.train_y, classifier.train_y_pred, title='XGBoost Train Confusion Matrix')
classifier.plot_decision_boundary(
    classifier.train_x, classifier.train_y, title='XGBoost Decision Boundary')
