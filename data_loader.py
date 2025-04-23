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

'''


import pandas as pd
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
        df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                      'hours-per-week', 'native-country', 'income']
        if output_file_path:
            df.to_csv(output_file_path, index=False)
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        预处理数据
        1. 处理缺失值
        2. 处理非数值
        3. 进一步处理
        '''
        # 处理缺失值——直接删
        df.replace(' ?', pd.NA, inplace=True)
        df.dropna(inplace=True)

        # 处理非数值
        # 
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

        return df


if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data('data/adult.csv')
    print(df.head())
