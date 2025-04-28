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
from evaluation import Evaluation


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

        print("accuracy on train set: ", accuracy_score(
            self.train_y, self.train_y_pred))
        print("accuracy on test set: ", accuracy_score(
            self.test_y, self.test_y_pred))
        print("accuracy on total set: ",
              accuracy_score(self.y, self.total_y_pred))


# 定义待测试的模型及标题
models = [
    ("SVM", SVC(kernel='linear', random_state=42)),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42)),
    ("XGBoost", XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        eta=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
]

evaluation = Evaluation()
# 使用SVM创建初始Classifier，后续通过set_model更换
classifier = Classifier(model=models[0][1], random_state=42)
classifier.load_and_preprocess_data('raw/adult.data')

# 遍历模型列表，依次训练、评估并绘制决策边界
for title, model in models:
    classifier.set_model(model)
    classifier.train()
    print(f"=== {title} Evaluation ===")
    evaluation.evaluate_model(
        classifier.model,
        classifier.train_x,
        classifier.train_y,
        classifier.test_x,
        classifier.test_y
    )
    evaluation.plot_decision_boundary(
        classifier.model,
        classifier.train_x,
        classifier.train_y,
        title=f'Decision Boundary of {title}'
    )
