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


from typing import Tuple
import os
import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, file_path: str, output_file_path=None) -> pd.DataFrame:
        '''
        读取csv或文本文件，支持单个文件或文件夹

        file_path: str, 文件路径或文件夹路径
        output_file_path: str, 输出文件路径, 如果不为None, 则将数据转为csv保存到该路径
        '''
        if os.path.isfile(file_path):
            # 如果是文件，直接读取
            df = pd.read_csv(file_path, header=None)
        elif os.path.isdir(file_path):
            # 如果是文件夹，读取所有文件并整合
            all_files = [os.path.join(file_path, f)
                         for f in os.listdir(file_path)]
            df_list = [pd.read_csv(f, header=None) for f in all_files]
            df = pd.concat(df_list, ignore_index=True)
        else:
            raise ValueError(f"{file_path} 不是有效的文件或文件夹路径")

        # 如果不是 CSV 文件，添加列名
        if not file_path.endswith('.csv'):
            df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income']

        # 如果指定了输出路径，保存为 CSV 文件
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

    def data_noise(self, df: pd.DataFrame, noise_level: float = 0.2) -> pd.DataFrame:
        """
        增加高斯噪音、随机clip和mask
        df: pd.DataFrame, 输入数据
        noise_level: float, 噪声强度
        return: pd.DataFrame, 添加噪声后的数据
        """
        # 增加高斯噪声
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] += np.random.normal(0, noise_level *
                                        df[col].std(), size=df[col].shape)

        # 随机clip
        for col in numeric_cols:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # # 随机mask
        # mask_prob = 0.01  # 1% 的概率将值设置为 NaN
        # mask = np.random.rand(*df.shape) < mask_prob
        # df = df.mask(mask)

        return df

    def preprocess_data(self, df: pd.DataFrame,
                        missing_value_method: str = "impute",
                        low_card_method: str = "onehot",
                        high_card_mehtod: str = "target",
                        if_standard: bool = True,
                        noise: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        预处理数据
        1. 处理缺失值
        2. 处理非数值
        3. 进一步处理

        df: pd.DataFrame, 原始数据
        missing_value_method: str, 缺失值处理方法, drop/impute
        low_card_method: str, 低基数特征处理方法, onehot/label/binary
        high_card_mehtod: str, 高基数特征处理方法, frequency/target/hashing
        return: Tuple[pd.DataFrame, pd.Series], 预处理后的数据和标签
        '''

        # 清洗
        df.columns = df.columns.str.replace('_', '.', regex=False)

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
                # 确保 'income' 列的值被正确处理
                df['income'] = df['income'].str.strip()  # 去除多余空格
                # 计算目标编码
                freq = df.groupby(col)['income'].value_counts(
                    normalize=True).unstack().fillna(0)
                # 检查列名是否为 '>50K' 或其他值
                if '>50K' in freq.columns:
                    df[col] = df[col].map(freq['>50K'])
                else:
                    raise ValueError(
                        f"Unexpected income column names in target encoding: {freq.columns}")
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
            X = pd.DataFrame(X_scaler, columns=X.columns, index=X.index)

        if noise:
            X = self.data_noise(X, 0.01)

        return X, y

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
    df = data_loader.load_data('raw')
    data_loader.check_missing_values(df)
    df, y = data_loader.preprocess_data(df)
    df = data_loader.save_data(df, 'data/train_preprocessed.csv')
    print(df.head())
    print(y.head())
    # 统计类别比例
    print("类别比例统计：")
    print(y.value_counts(normalize=True))
    print("y中含有NA值：", y.isna().any())
    print("NA值数量：", y.isna().sum())
