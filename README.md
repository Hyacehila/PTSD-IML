# PTSD-IML: 基于机器学习的PTSD预测研究项目

## 项目概述

本项目是一个完整的PTSD（创伤后应激障碍）预测机器学习研究项目，通过分析多种临床和人口统计学特征，构建随机森林模型来预测PCL（PTSD检查表）总分。项目包含完整的数据预处理、特征工程、模型训练、性能评估和可解释性分析流程。

## 项目结构

```
PTSD-IML/
├── data/                           # 数据文件目录
│   ├── PTSD.xlsx                  # 原始Excel数据集
│   ├── data.csv                   # 预处理后的CSV格式数据
│   └── data.pkl                   # 序列化的Python数据文件
├── models/                        # 模型文件目录
│   └── final_random_forest_model.pkl  # 训练完成的随机森林模型
├── notebooks/                     # Jupyter分析笔记本
│   ├── data_pre.ipynb            # 数据预处理和特征工程
│   └── prediction.ipynb          # 模型训练、预测和SHAP分析
├── README.md                      # 项目说明文档
├── LICENSE                        # 开源许可证
└── .gitignore                     # Git版本控制忽略文件
```

## 数据集详情

### 原始数据特征

项目使用的数据集包含以下16个核心特征：

#### 人口统计学特征
- **年龄** (age): 连续数值变量
- **性别** (gender): 分类变量
- **文化程度** (education): 有序分类变量 (3=高中及以下, 2=大专, 1=本科及以上)
- **婚姻状况** (marital_status): 分类变量
- **家庭人均月收入** (income): 有序分类变量 (1=低, 2=中低, 3=中高, 4=高)
- **BMI** (bmi): 连续数值变量

#### 创伤暴露特征
- **近期是否出现意外事件** (recent_accident): 二分类变量
- **是否有家庭成员或亲近朋友意外** (family_accident): 二分类变量
- **意外事件中是否目睹有人重伤** (witnessed_injury): 二分类变量
- **意外事件及救援过程中，您是否接触或见过尸体** (witnessed_corpse): 二分类变量
- **您是否曾因为灾难现场的景象** (traumatic_scene): 二分类变量

#### 临床特征
- **ASD性质** (asd_nature): 二分类变量
- **既往是否有精神病史** (psychiatric_history): 二分类变量
- **近一月内是否使用药物** (medication): 二分类变量
- **是否吸烟（包括以前）** (smoking): 二分类变量
- **是否饮酒** (alcohol): 二分类变量
- **心理韧性** (resilience): 连续数值变量

#### 目标变量
- **PCL总分** (pcl_scores): 连续数值变量，范围0-80

### 数据预处理流程

1. **数据清洗**
   - 删除无意义列：序号、籍贯、民族
   - 删除重复行
   - 检查和处理缺失值

2. **特征选择**
   - 基于Pearson相关系数分析，移除与PCL总分相关性不显著的变量
   - 移除身高、体重（相关系数<0.1, p>0.05）
   - 保留BMI（基于相关研究支持）
   - 移除ASD各子项分数（避免与ASD总分的多重共线性）

3. **数据类型转换**
   - 数值变量：保持原始类型
   - 有序分类变量：文化程度、家庭人均月收入
   - 名义分类变量：其他所有分类变量

4. **特征编码**
   - 数值特征：标准化处理
   - 有序特征：OrdinalEncoder编码
   - 名义特征：OneHotEncoder编码

## 机器学习模型

### 随机森林回归器

使用经过超参数优化的随机森林模型：

```python
RandomForestRegressor(
    n_estimators=495,           # 树的数量
    max_features=0.540881,      # 最大特征比例
    min_samples_leaf=42,        # 叶节点最小样本数
    n_jobs=1,                   # 并行作业数
    random_state=42             # 随机种子
)
```

### 模型性能指标

- **训练集RMSE**: 均方根误差（具体数值需运行prediction.ipynb查看）
- **训练集MAE**: 平均绝对误差（具体数值需运行prediction.ipynb查看）

### 特征工程管道

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('ord', OrdinalEncoder(), ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 list(set(categorical_cols) - set(ordinal_cols)))
    ]
)
```

## 模型可解释性

### SHAP分析

项目使用SHAP（SHapley Additive exPlanations）进行模型可解释性分析：

1. **SHAP特征重要性图**
   - 显示各特征对模型预测的平均影响
   - 使用蓝色渐变色彩，重要性越高颜色越深
   - 提供具体的SHAP值数值

2. **SHAP摘要图**
   - 展示特征值对预测的影响分布
   - 显示特征值的高低如何影响PCL总分预测
   - 识别特征与目标变量的非线性关系

### 特征映射处理

由于OneHot编码会产生多个衍生特征，项目实现了智能的特征映射算法：
- 数值特征：1对1映射
- 有序特征：1对1映射  
- 分类特征：1对多映射，合并SHAP值

## 统计分析

### 相关性分析

1. **Pearson相关分析**
   - 分析数值变量与PCL总分的线性相关性
   - 提供相关系数、t统计量和p值
   - 显著性标记：***p<0.001, **p<0.01, *p<0.05

2. **点二列相关分析**
   - 分析二分类变量与连续变量的相关性
   - 应用于ASD性质、遗传史、性别等变量

3. **关联矩阵分析**
   - 使用Cramér's V分析分类变量间关联
   - 使用Correlation Ratio分析分类与数值变量关联
   - 生成完整的关联矩阵热力图

## 使用方法

### 环境要求

```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
openpyxl>=3.0.0
joblib>=1.1.0
scipy>=1.7.0
```

### 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap openpyxl joblib scipy
```

### 运行流程

1. **数据预处理**
   ```bash
   jupyter notebook notebooks/data_pre.ipynb
   ```
   - 加载原始Excel数据
   - 执行数据清洗和特征选择
   - 生成预处理后的数据文件

2. **模型训练和分析**
   ```bash
   jupyter notebook notebooks/prediction.ipynb
   ```
   - 加载预处理数据
   - 训练随机森林模型
   - 执行SHAP可解释性分析
   - 生成特征重要性图表

### 数据文件路径说明

- 原始数据：`data/PTSD.xlsx`
- 预处理后数据：`data/data.pkl` (推荐使用) 或 `data/data.csv`
- 训练好的模型：`models/final_random_forest_model.pkl`

## 主要发现

### 关键预测特征

通过SHAP分析识别出对PTSD预测最重要的特征（按重要性排序）：
1. **心理韧性** - 最重要的影响因素
2. **年龄** - 重要的人口统计学因素
3. **BMI** - 身体健康指标
4. **创伤暴露相关特征** - 意外事件经历
5. **临床特征** - 精神病史、用药情况等

### 模型特点

- **高可解释性**：通过SHAP提供透明的预测解释
- **鲁棒性强**：随机森林对异常值和噪声具有较好的鲁棒性
- **特征重要性明确**：清晰识别各特征的相对重要性

## 技术亮点

1. **完整的数据科学流程**：从原始数据到最终模型的端到端处理
2. **先进的可解释性分析**：使用SHAP提供模型透明度
3. **智能的特征工程**：自动化的特征类型识别和编码
4. **统计严谨性**：全面的相关性分析和显著性检验
5. **可视化丰富**：多种图表展示数据关系和模型结果

## 注意事项

1. **研究用途**：本项目仅用于学术研究，不应用于临床诊断
2. **数据隐私**：使用时请遵守相关数据使用协议和隐私保护规定
3. **模型泛化**：模型性能基于特定数据集，在新数据上可能需要重新验证
4. **依赖版本**：建议使用指定版本的Python包以确保兼容性

## 贡献指南

欢迎通过以下方式参与项目：
- 提交Issue报告问题或建议
- 提交Pull Request改进代码或文档
- 分享使用经验和应用案例

## 许可证

本项目采用MIT开源许可证，详见[LICENSE](LICENSE)文件。

## 引用

如果您在研究中使用了本项目，请引用：

```
PTSD-IML: 基于机器学习的PTSD预测研究项目
GitHub仓库: [您的仓库链接]
```

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
