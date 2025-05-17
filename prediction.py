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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_loader import DataLoader
from evaluation import Evaluation
from sklearn.model_selection import RandomizedSearchCV
import time



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

    def random_search(self, param_grid, n_iter=5):
        """
        随机搜索超参数
        """
        # 定义随机搜索
        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=2,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2,
        )
        search.fit(self.train_x, self.train_y)
        print("Best parameters found: ", search.best_params_)
        return search.best_estimator_, search.best_params_


# 3. XGBoost（二分类）
param_grid_xgb = {
    # 学习率
    'eta': [0.01, 0.05, 0.1, 0.2],
    # 树的最大深度
    'max_depth': [3, 5, 7, 10],
    # 样本采样比例
    'subsample': [0.6, 0.8, 1.0],
    # 列采样比例
    'colsample_bytree': [0.6, 0.8, 1.0],
    # 树间最小损失减少，用于控制过拟合
    'gamma': [0, 0.1, 0.2, 0.5],
    # L1 正则化项
    'alpha': [0, 0.1, 1, 10],
    # L2 正则化项
    'lambda': [1, 2, 5, 10]
}

# 定义待测试的模型及标题
models = [
    ("Logistic Regression", LogisticRegression(
        max_iter=1000, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(
        max_depth=6, random_state=42)),
    ("SVM", SVC(kernel='linear', random_state=42)),
    # ("Random Forest", RandomForestClassifier(
    #     n_estimators=100, max_depth=6, random_state=42)),
    # ("XGBoost", XGBClassifier(
    #     objective='binary:logistic',
    #     eval_metric='logloss',
    #     eta=0.1,
    #     max_depth=6,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )),
    # ("KNN", KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)),
    # ("MLP", MLPClassifier(
    #     hidden_layer_sizes=(100,),  # 隐藏层大小（100 个神经元）
    #     activation='relu',         # 激活函数（默认 relu）
    #     solver='adam',             # 优化器（默认 adam）
    #     max_iter=300,              # 最大迭代次数
    #     random_state=42
    # )),
    # ("LGBM", LGBMClassifier(
    #     objective='binary',
    #     metric='binary_logloss',
    #     learning_rate=0.1,
    #     max_depth=6,
    #     num_leaves=31,
    #     min_child_samples=10,
    #     min_split_gain=0.01,
    #     max_bin=128,
    #     class_weight='balanced',
    #     random_state=42
    # )),
    # ("CatBoost", CatBoostClassifier(
    #     iterations=1000,
    #     depth=6,

    #     learning_rate=0.1,
    #     loss_function='Logloss',
    #     random_seed=42,
    #     verbose=0
    # ))
]


def main_train_test():
    evaluation = Evaluation()
    # 使用第一个模型初始化 Classifier
    classifier = Classifier(model=models[0][1], random_state=42)
    classifier.load_and_preprocess_data('raw')

    # 遍历模型列表，依次训练、评估
    for title, model in models:
        print(f"=== {title} Evaluation ===")
        classifier.set_model(model)
        start_time = time.time()
        classifier.train()
        end_time = time.time()
        evaluation.evaluate_model(
            classifier.model,
            classifier.train_x,
            classifier.train_y,
            classifier.test_x,
            classifier.test_y
        )
        print(f"Training time: {end_time - start_time:.2f} seconds")

        # print("=== Cross Validation ===")
        # start_time = time.time()
        # try:
        #     # 确保模型兼容 cross_val_score
        #     if hasattr(model, "fit") and hasattr(model, "predict"):
        #         scores = cross_val_score(
        #             classifier.model, classifier.x, classifier.y, cv=5
        #         )
        #         print(f"Cross-validation scores: {scores}")
        #         print(f"Mean cross-validation score: {scores.mean():.4f}")
        #     else:
        #         print(f"{title} is not compatible with cross_val_score.")
        # except Exception as e:
        #     print(f"Error during cross-validation for {title}: {e}")
        # end_time = time.time()
        # print(f"Cross-validation time: {end_time - start_time:.2f} seconds")


def main_random_search():
    # 使用SVM创建初始Classifier，后续通过set_model更换
    classifier = Classifier(model=models[0][1], random_state=42)
    classifier.load_and_preprocess_data('raw/adult.data')

    # XGBoost
    classifier.set_model(PatchedXGBClassifier(random_state=42))
    best_model, best_param = classifier.random_search(param_grid_xgb, n_iter=3)
    classifier.set_model(best_model)
    classifier.train()
    print("=== Best XGBoost Model Evaluation ===")
    print("Best parameters: ", best_param)
    classifier.evaluate()


if __name__ == "__main__":
    # main_train_test()
    main_random_search()