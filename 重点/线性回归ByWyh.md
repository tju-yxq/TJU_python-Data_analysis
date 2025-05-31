### 一、线性回归原理

#### 1. 概述

线性回归是一种统计方法，用于确定两个或多个变量之间的线性关系。它通过找出一条**最佳拟合直线**来描述因变量（Y）如何随一个或多个自变量（X）的变化而变化。

- **简单线性回归公式：** Y=a+b⋅X  
    
    - Y: 因变量的预测值
        
    - a: 回归线在Y轴上的截距（当X=0时，Y的期望值）
        
    - b: 回归系数（X每增加一个单位，Y预期的平均变化量）
        
    - X: 自变量
        
- **多元线性回归公式：** y=β0​+β1​x1​+β2​x2​+⋅⋅⋅+βp​xp​+ϵ  
    
    - 扩展到多个自变量，βi​ 为对应自变量的系数，ϵ 为误差项。
        
- **矩阵表示法：** Y=Xβ+ϵ  
    

#### 2. 核心假设

为确保线性回归的有效性和结果可靠性，需满足以下假设：

- **线性关系：** 自变量和因变量之间存在线性关系。
    
- **独立性：** 观测值之间相互独立。
    
- **同方差性 (Homoscedasticity)：** 误差项的方差在所有自变量水平上是相同的。
    
- **正态性：** 误差项 ϵ 服从均值为0、标准差固定的正态分布。
    
- **无多重共线性：** 在多元回归中，自变量间不存在严格的线性关系。
    

#### 3. 最小二乘法 (Least Squares Method)

- **定义：** 一种通过最小化预测值与实际值之间的**残差平方和 (RSS)** 来确定模型参数的最优解的数学方法。
    
- **核心思想：** 找到一组参数，使得模型预测结果与真实数据之间的误差尽可能小。
    
- **残差平方和 (RSS)：** RSS=∑i=1n​(yi​−(β0​+β1​xi​))2  
    
- **求解：** 通过对RSS求偏导并令其为0，解得 β0​ 和 β1​ 的最优解。
    
    - β1​=∑i=1n​(xi​−x)2∑i=1n​(xi​−x)(yi​−y​)​  
        
    - β0​=y​−β1​x  
        
- **多元线性回归的矩阵解：** β=(X⊤X)−1X⊤Y  
    

### 二、线性回归的实现 (基于Sklearn)

#### 1. Sklearn 简介

- **定义：** Scikit-learn (Sklearn) 是一个广泛使用的Python机器学习库，提供简单高效的工具。
    
- **特点：** 易于使用、开源、文档丰富、社区支持。
    
- **常用模块：** 分类、回归、聚类、降维、模型选择、预处理等。
    

#### 2. 核心函数

- **导入模型：** `from sklearn.linear_model import LinearRegression`
    
- **创建模型实例：** `model = LinearRegression()`
    
- **训练模型：** `model.fit(X, y)`
    
- **进行预测：** `model.predict(X_test)`
    
- **获取系数和截距：** `model.coef_`, `model.intercept_`
    
- **数据划分：** `from sklearn.model_selection import train_test_split`
    
    - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)`
        
    - **参数：** `*arrays` (输入数据), `test_size` (测试集比例), `train_size` (训练集比例), `random_state` (随机种子), `shuffle` (是否打乱), `stratify` (分层采样)。
        
- **数据标准化：** `from sklearn.preprocessing import StandardScaler`
    
    - `scaler = StandardScaler()`
        
    - `X_scaled = scaler.fit_transform(X)`
        
    - **标准化公式：** z=σx−μ​  

### 三、模型的评估指标

#### 1. 回归任务常用指标

- **MSE (均方误差)：** MSE=n1​∑i=1n​(yi​−y^​i​)2  
    
    - **优点：** 对大误差惩罚大，数学性质好。
        
    - **缺点：** 对异常值敏感。
        
- **RMSE (均方根误差)：** RMSE=MSE​  
    
    - **优点：** 与因变量量纲一致，更直观。
        
    - **缺点：** 同样对异常值敏感。
        
- **MAE (平均绝对误差)：** MAE=n1​∑i=1n​∣yi​−y^​i​∣  
    
    - **优点：** 对异常值鲁棒性强。
        
    - **缺点：** 对大误差惩罚较弱。
        
- **R²系数 (决定系数)：** R2=1−TSSRSS​ (TSS为总平方和)
    
    - **优点：** 反映模型拟合度，越接近1越好。
        
    - **缺点：** 增加不相关特征也可能提高R²。
        
- **调整R² (Adjusted R²):** 解决了R²因特征数量膨胀的问题。
    
    - **优点：** 考虑样本量和特征数量影响，更适合多特征模型。
        
    - **缺点：** 解释性较弱，需配合其他指标。
        

#### 2. 其他常用评估方法

- **残差分析：** 检查残差是否符合正态分布，判断模型拟合情况。
    
- **K折交叉验证 (K-Fold Cross-Validation)：** `from sklearn.model_selection import cross_val_score`
    
    - 将数据集分为K个子集，轮流作为验证集，其余作为训练集，取平均结果。
        
    - **优点：** 避免数据划分导致的过拟合，尤其适合小数据集。
        
- **学习曲线 (Learning Curve)：** 诊断模型过拟合或欠拟合。
    
    - **过拟合：** 训练误差低，验证误差高，且不随样本增加而下降。
        
    - **欠拟合：** 训练误差和验证误差都高，且不随样本增加而下降。
        
- **模型复杂度 vs 泛化误差：** 通过图表选择合适的模型复杂度。
    
- **早停法 (Early Stopping)：** 训练过程中监测验证集性能，当不再改进时停止训练，防止过拟合。
    

### 四、Python库应用

- **数据处理：** `pandas`, `numpy`
    
- **模型构建：** `sklearn.linear_model` (如 `LinearRegression`)
    
- **数据预处理：** `sklearn.preprocessing` (如 `StandardScaler`)
    
- **模型选择与评估：** `sklearn.model_selection` (如 `train_test_split`, `cross_val_score`), `sklearn.metrics` (用于计算MSE, MAE等)
    
- **可视化：** `seaborn`, `matplotlib.pyplot`
