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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from data_loader import DataLoader
from evaluation import Evaluation
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform,randint


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
        return search.best_estimator_,search.best_params_




# 1. SVM (支持向量机)
param_grid_svm = {
    # 惯用对数空间，尝试从 0.001 到 100 的 C
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    # 核函数：线性、RBF、多项式
    'kernel': ['linear', 'rbf', 'poly'],
    # 只有在 kernel='rbf' 或 'poly' 时才生效
    'gamma': [1e-3, 1e-2, 1e-1, 1, 10],
    # 多项式核时的阶数
    'degree': [2, 3, 4]
}

# 2. Random Forest（随机森林）
param_grid_rf = {
    # 树的数量
    'n_estimators': [50, 100, 200, 500],
    # 最大深度
    'max_depth': [None, 5, 10, 20, 50],
    # 每次分裂所需的最小样本数
    'min_samples_split': [2, 5, 10],
    # 叶子节点最小样本数
    'min_samples_leaf': [1, 2, 4],
    # 考虑特征的最大数量（比例或绝对值）
    'max_features': ['auto', 'sqrt', 0.2, 0.5]
}

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

def main_train_test():
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
        # evaluation.plot_decision_boundary(
        #     classifier.model,
        #     classifier.train_x,
        #     classifier.train_y,
        #     title=f'Decision Boundary of {title}'
        # )

def main_random_search():
    # 使用SVM创建初始Classifier，后续通过set_model更换
    classifier = Classifier(model=models[0][1], random_state=42)
    classifier.load_and_preprocess_data('raw/adult.data')

    # 随机搜索超参数
    # SVM
    best_model,best_param = classifier.random_search(param_grid_svm, n_iter=3)
    classifier.set_model(best_model)
    classifier.train()
    print("=== Best SVM Model Evaluation ===")
    print("Best parameters: ", best_param)
    classifier.evaluate()
    
    # 随机森林
    classifier.set_model(RandomForestClassifier(random_state=42))
    best_model,best_param = classifier.random_search(param_grid_rf, n_iter=3)
    classifier.set_model(best_model)
    classifier.train()
    print("=== Best Random Forest Model Evaluation ===")
    print("Best parameters: ", best_param)
    classifier.evaluate()
    
    # XGBoost   
    classifier.set_model(XGBClassifier(random_state=42))
    best_model,best_param = classifier.random_search(param_grid_xgb, n_iter=3)
    classifier.set_model(best_model)
    classifier.train()
    print("=== Best XGBoost Model Evaluation ===")
    print("Best parameters: ", best_param)
    classifier.evaluate()

if __name__ == "__main__":
    # main_train_test()
    main_train_test()