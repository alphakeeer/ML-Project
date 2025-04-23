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


import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

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

    def preprocess_data_svm(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        '''
        预处理数据
        1. 处理缺失值
        2. 处理非数值
        3. 进一步处理
        '''

        # 处理缺失值->直接删
        df.replace(' ?', pd.NA, inplace=True)
        df.dropna(inplace=True)

        # 处理非数值
        low_card = ['sex', 'relationship', 'race', 'marital-status']
        high_card = ['workclass', 'education', 'occupation', 'native-country']

        # 低基数->onehont
        df = pd.get_dummies(df, columns=low_card, drop_first=True)

        # 高基数->频率编码
        for col in high_card:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
        
        # 分离标签
        y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        X = df.drop('income', axis=1)
        
        # 标准化
        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)
        X_df=pd.DataFrame(X_scaler, columns=X.columns,index=X.index)

        return X_df, y


if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data('raw/adult.data', 'data/adult.csv')
    df,y = data_loader.preprocess_data_svm(df)
    df=data_loader.save_data(df, 'data/adult_preprocessed.csv')
    print(df.head())
