'''
1. Examine and Handle missing values (e.g., fill the missing value, add a corresponding label).
2. Handle non-numeric values (e.g. one-hot encoding, Boolean indicator).
3. Further processing (e.g. standardize features).
'''
import pandas as pd
import numpy as np
import os

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

def handle_missing_values():
    pass

def csv_convert(input_path, output_path):
    """
    将数据文件转换为 CSV 格式
    """
    try:
        # 读取数据文件
        df = pd.read_csv(input_path, header=None, names=columns, skipinitialspace=True)
        # 保存为 CSV 文件
        df.to_csv(output_path, index=False)
        print(f"成功将 {input_path} 转换为 {output_path}")
    except Exception as e:
        print(f"转换失败: {e}")

def missing_value_processing(input_path, output_path):
    """
    处理缺失值
    """
    # 需要避开第一行，因为它是列名
    try:
        # 读取数据文件
        df = pd.read_csv(input_path, header=None, names=columns, skipinitialspace=True)
        # 将缺失值替换为 NaN
        df.replace('?', pd.NA, inplace=True)
        
        # 对于数值变量，使用均值填充缺失值
        numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)

        # 对于分类变量，使用众数填充缺失值
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
        for col in categorical_columns:
            mode_value = df[col].mode()[0]
            print(f"Mode value for {col}: {mode_value}")
            df[col] = df[col].fillna(mode_value)
        
        # 保存处理后的数据
        df.to_csv(output_path, index=False)
        print(f"成功处理 {input_path} 的缺失值并保存为 {output_path}")

    except Exception as e:
        print(f"处理{input_path}缺失值失败: {e}")
    
    

if __name__ == "__main__":
    # Load the dataset
    train_data_path = './raw/adult.data'
    test_data_path = './raw/adult.test'

    # 如果数据集存在，输出提示信息
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print("数据集路径已存在，开始处理数据集...")
    else:
        print("非法数据集路径，请检查路径合法性。")
        exit()

    # 读取数据集并转换成csv格式
    train_csv_path = './data/adult_train.csv'
    test_csv_path = './data/adult_test.csv'
    csv_convert(train_data_path, train_csv_path)
    csv_convert(test_data_path, test_csv_path)

    # 1.对于训练集和测试集，先进行缺失值处理
    # 此处缺失值表示为 "?"；对于数值变量，使用均值填充；对于分类变量，使用众数填充
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    new_train_csv_path = './data/adult_train_processed.csv'
    new_test_csv_path = './data/adult_test_processed.csv'
    missing_value_processing(train_csv_path, new_train_csv_path)
    missing_value_processing(test_csv_path, new_test_csv_path)
    


