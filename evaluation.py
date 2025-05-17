'''
Objective: Analyze and improve the models' performance.
Instructions
    1. Calculate metrics such as accuracy, precision, recall, and F1-score for each model trained in Part 3 (using confusion matrix).
    2. Draw ROC and calculate AUC for each model class.
    3. Improve each model via validation.
    4. Interpret the results to assess each model's strengths, weaknesses and possible improvements (e.g., determining whether the model overfits).
Deliverables
1. Calculated metrics, e.g., AUC and plotted ROC.
2. Description and interpretation of validation results.
3. Further discussion (100-200 words) of model performance based on metrics, ROC, AUC and so on.
'''

from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import numpy as np


class Evaluation:
    def __init__(self):
        pass

    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        """
        绘制混淆矩阵，同时显示每个单元格的计数及百分比，并调整字体大小
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        total = np.sum(cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 设置坐标轴标签和刻度字体大小
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xlabel='Predicted',
            ylabel='True',
            title=title
        )
        ax.tick_params(axis='both', labelsize=14)  # 调整刻度文字大小
        
        # 在每个格子上标注计数和百分比，并设置字体大小
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total
                ax.text(j, i, f'{cm[i, j]}\n({percentage:.2%})',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=26)
        
        plt.tight_layout()
        plt.show()
        
    def plot_decision_boundary(self, model, x, y, title):
        """
        绘制决策边界
        """
        # 使用 PCA 降维到 2D
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(x)
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor='k')
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    def plot_roc_curve(self, y_true, y_scores, title='ROC Curve'):
        """
        绘制ROC曲线
        """
        # SVM 的得分：decision_function；其他模型可用 predict_proba
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def classification_report(self, y_true, y_pred):
        """
        打印分类报告
        """
        report = classification_report(y_true, y_pred)
        print(report)
        return report

    def evaluate_model(self, model, x_train, y_train, x_test, y_test):
        """
        评估模型性能
        """
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # 计算准确率
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # 绘制混淆矩阵
        self.plot_confusion_matrix(
            y_train, y_train_pred, title='Train Confusion Matrix')
        self.plot_confusion_matrix(
            y_test, y_test_pred, title='Test Confusion Matrix')
        # 绘制总体混淆矩阵
        self.plot_confusion_matrix(
            np.concatenate((y_train, y_test)), np.concatenate((y_train_pred, y_test_pred)), title='Total Confusion Matrix')

        # # 绘制ROC曲线
        # if hasattr(model, "predict_proba"):
        #     y_train_scores = model.predict_proba(x_train)[:, 1]
        #     y_test_scores = model.predict_proba(x_test)[:, 1]
        #     self.plot_roc_curve(y_train, y_train_scores,
        #                         title='Train ROC Curve')
        #     self.plot_roc_curve(y_test, y_test_scores, title='Test ROC Curve')
        #     # 绘制总体ROC曲线
        #     y_total_scores = model.predict_proba(
        #         np.concatenate((x_train, x_test)))[:, 1]
        #     self.plot_roc_curve(np.concatenate((y_train, y_test)),
        #                         y_total_scores, title='Total ROC Curve')
        # elif hasattr(model, "decision_function"):
        #     y_train_scores = model.decision_function(x_train)
        #     y_test_scores = model.decision_function(x_test)
        #     self.plot_roc_curve(y_train, y_train_scores,
        #                         title='Train ROC Curve')
        #     self.plot_roc_curve(y_test, y_test_scores, title='Test ROC Curve')
        #     # 绘制总体ROC曲线
        #     y_total_scores = model.decision_function(
        #         np.concatenate((x_train, x_test)))
        #     self.plot_roc_curve(np.concatenate((y_train, y_test)),
        #                         y_total_scores, title='Total ROC Curve')

        # 打印分类报告
        print("Classification Report:")
        print("Train Set:")
        self.classification_report(y_train, y_train_pred)
        print("Test Set:")
        self.classification_report(y_test, y_test_pred)
        print("Total Set:")
        self.classification_report(
            np.concatenate((y_train, y_test)), np.concatenate((y_train_pred, y_test_pred)))
