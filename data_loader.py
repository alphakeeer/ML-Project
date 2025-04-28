'''
1. 处理缺失值
2. 处理非数值
3. 进一步处理

读取数据集并返回所需类型
'''

'''
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

>50K, <=50K.

workclass: 8
education: 16
marital-status: 7
occupation: 14
relationship: 6
race: 5
sex: 2
native-country: 41
'''

'''
SVM适合数据：
1. 高纬稠密特征
2. 线性可分
3. 需要归一化/标准化

低纬度——onehot
高纬度——降维/特征选择
'''

'''
xgboost适合数据：
1. 数值/稀疏混合
2. 非线性关系/交互
3. 允许部分缺失

缺失值标记——NaN
类别特征——低基类onehot，高基类目标编码
'''


from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import category_encoders as ce
from typing import Tuple


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, file_path: str, output_file_path=None) -> pd.DataFrame:
        '''
        读取csv或文本文件

        file_path: str, 文件路径
        output_file_path: str, 输出文件路径, 如果不为None, 则将数据转为csv保存到该路径
        '''
        df = pd.read_csv(file_path, header=None)
        if not file_path.endswith('.csv'):
            df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income']
        if output_file_path:
            df.to_csv(output_file_path, index=False)
        return df

    def save_data(self, data, file_path: str):
        '''
        保存数据到csv文件
        data: pd.DataFrame 或 numpy.ndarray, 数据
        file_path: str, 文件路径
        '''
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        data.to_csv(file_path, index=False)
        return data

    def preprocess_data(self, df: pd.DataFrame,
                        missing_value_method: str = "drop",
                        low_card_method: str = "onehot",
                        high_card_mehtod: str = "frequency",
                        if_standard: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        预处理数据
        1. 处理缺失值
        2. 处理非数值
        3. 进一步处理

        df: pd.DataFrame, 原始数据
        missing_value_method: str, 缺失值处理方法, drop/impute
        low_card_method: str, 低基数特征处理方法, onehot/label/binary
        high_card_mehtod: str, 高基数特征处理方法, frequency/target
        return: Tuple[pd.DataFrame, pd.Series], 预处理后的数据和标签
        '''

        if missing_value_method == "drop":
            # 处理缺失值->直接删
            df.replace(' ?', pd.NA, inplace=True)
            df.dropna(inplace=True)
        elif missing_value_method == "impute":
            # 处理缺失值->插补，类别用众数，数值用中位数
            numeric_cols = ['age', 'fnlwgt', 'education-num',
                            'capital-gain', 'capital-loss', 'hours-per-week']
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'native-country']

            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        # 处理非数值
        low_card = ['sex', 'relationship', 'race', 'marital-status']
        high_card = ['workclass', 'education', 'occupation', 'native-country']

        if low_card_method == "onehot":
            # 低基数->onehot
            df = pd.get_dummies(df, columns=low_card, drop_first=True)
        elif low_card_method == "label":
            # 低基数->label encoding
            for col in low_card:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif low_card_method == "binary":
            # 低基数->Binary encoding
            encoder = ce.BinaryEncoder(cols=low_card)
            df = encoder.fit_transform(df)
        else:
            raise ValueError("low_card_method must be onehot, label or binary")

        if high_card_mehtod == "frequency":
            # 高基数->频率编码
            for col in high_card:
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
        elif high_card_mehtod == "target":
            # 高基数->目标编码
            for col in high_card:
                freq = df.groupby(col)['income'].value_counts(
                    normalize=True).unstack().fillna(0)
                freq = freq.div(freq.sum(axis=1), axis=0)
                df[col] = df[col].map(freq[1])
        elif high_card_mehtod == "hashing":
            # 高基数->Hashing encoding
            encoder = ce.HashingEncoder(cols=high_card, n_components=8)
            df = encoder.fit_transform(df)

        else:
            raise ValueError("high_card_mehtod must be frequency or target")
        # 分离标签
        y = df['income'].str.strip().map({'>50K': 1, '<=50K': 0})
        X = df.drop('income', axis=1)

        if if_standard:
            # 标准化
            scaler = StandardScaler()
            X_scaler = scaler.fit_transform(X)
            X_df = pd.DataFrame(X_scaler, columns=X.columns, index=X.index)

        return X_df, y

    def check_missing_values(self, df: pd.DataFrame):
        '''
        检测缺失值
        df: pd.DataFrame, 原始数据
        '''

        df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                      'hours-per-week', 'native-country', 'income']

        # 替换表示缺失值的字符串
        df.replace(' ?', pd.NA, inplace=True)

        # 查看缺失值情况
        print("缺失值统计：")
        print(df.isna().sum())

        # 方法1：直接删除缺失值的样本
        df_dropna = df.dropna()

        # 方法2：针对类别变量使用众数插补，数值变量使用中位数插补（示例）
        df_impute = df.copy()
        numeric_cols = ['age', 'fnlwgt', 'education-num',
                        'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country', 'income']

        for col in numeric_cols:
            df_impute[col] = df_impute[col].fillna(df_impute[col].median())

        for col in categorical_cols:
            df_impute[col] = df_impute[col].fillna(df_impute[col].mode()[0])

        # 对比处理结果
        print("删除缺失值后样本量：", df_dropna.shape[0])
        print("插补后样本量：", df_impute.shape[0])


if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data('raw/adult.data')
    # data_loader.check_missing_values(df)
    df, y = data_loader.preprocess_data(df)
    df = data_loader.save_data(df, 'data/train_preprocessed.csv')
    print(df.head())
    print(y.head())
