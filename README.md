# ML-Project
## 文件安排
- 文件架构
```
├── README.md
├── clustering.ipynb
├── clustering.py           # 聚类实现
├── data_loader.py          # 数据读取
├── evaluation.py           # 用于prediction的评估
├── prediction.ipynb        
├── prediction.py           # 模型训练和预测
├── preprocess.py           # 预处理数据
├── processing_output       # 文件夹：存储执行过程中非最终结果的文件，如训练中的模型
├── raw                     # 文件夹：原始数据集，这里面的别修改
│   ├── adult.data
│   ├── adult.names
│   └── adult.test
├── result_output           # 文件夹：结果的输出放这如：可视化的图
└── visualize.py            # 可视化文件
```
- ipynb文件最后整合用，其他文件如果要互相调用的话使用 `from data_loader import load_csv`这样子
- 每一个文件的具体要求请参考pdf