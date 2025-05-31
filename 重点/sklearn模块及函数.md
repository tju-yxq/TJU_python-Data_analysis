## Sklearn 常用模块及函数复习笔记

### 一、Sklearn 概述

**Scikit-learn (Sklearn)** 是一个基于 Python 的开源机器学习库，提供了大量用于数据挖掘和数据分析的简单高效工具。它构建在 NumPy、SciPy 和 Matplotlib 之上，是进行机器学习任务的强大平台。

- **特点：** 易于使用、高效、功能全面、文档丰富。
    
- **主要功能：** 分类、回归、聚类、降维、模型选择、预处理。
    

### 二、常用模块及核心函数

#### 1. 数据预处理 (Preprocessing)

用于数据清洗、转换和特征工程。

- `sklearn.preprocessing`:
    
    - `StandardScaler()`: **标准化**数据（均值0，方差1）。
        
        - `scaler.fit_transform(X)`
            
        - **公式：** z=σx−μ​  
            
    - `MinMaxScaler()`: **归一化**数据到指定范围（通常是 [0,1]）。
        
        - `scaler.fit_transform(X)`
            
        - **公式：** Xnorm​=Xmax​−Xmin​X−Xmin​​  
            
    - `OneHotEncoder()`: **独热编码**分类特征。
        
        - `encoder.fit_transform(categorical_data)`
            
    - `LabelEncoder()`: **标签编码**目标变量或分类特征（将类别映射为整数）。
        
        - `encoder.fit_transform(labels)`
            
    - `PolynomialFeatures()`: 生成**多项式特征**，用于捕捉非线性关系。
        
        - `poly = PolynomialFeatures(degree=2)`
            
        - `X_poly = poly.fit_transform(X)`
            
- `sklearn.impute.SimpleImputer()`: **填充缺失值**（均值、中位数、众数等）。
    
    - `imputer = SimpleImputer(strategy='mean')`
        
    - `X_imputed = imputer.fit_transform(X)`
        

#### 2. 数据划分与模型选择 (Model Selection)

用于数据集的划分、交叉验证和模型参数调优。

- `sklearn.model_selection`:
    
    - `train_test_split()`: **划分数据集**为训练集和测试集。
        
        - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
            
    - `cross_val_score()`: 进行**K折交叉验证**，评估模型性能。
        
        - `scores = cross_val_score(model, X, y, cv=5)`
            
    - `GridSearchCV()`: **网格搜索**，用于自动调优模型超参数。
        
        - `grid_search = GridSearchCV(estimator, param_grid, cv=5)`
            
        - `grid_search.fit(X, y)`
            
        - `grid_search.best_params_`, `grid_search.best_score_`
            

#### 3. 回归模型 (Regression Models) - **重点**

用于预测连续值的模型。

- `sklearn.linear_model`:
    
    - `LinearRegression()`: **普通最小二乘线性回归**。
        
        - **特点：** 简单、可解释性强、计算快。
            
        - **缺点：** 对异常值敏感，要求数据呈线性关系。
            
    - `Ridge()`: **岭回归**（L2正则化），用于处理多重共线性和过拟合。
        
        - **特点：** 惩罚大系数，使模型更稳定。
            
        - **缺点：** 会将系数缩小到接近零，但不会完全为零。
            
    - `Lasso()`: **Lasso回归**（L1正则化），用于特征选择和处理过拟合。
        
        - **特点：** 可以使部分系数**完全为零**，实现特征选择。
            
        - **缺点：** 对共线性特征的处理可能不稳定。
            
    - `ElasticNet()`: **弹性网络回归**（L1和L2正则化结合）。
        
        - **特点：** 结合Lasso和Ridge的优点，既能特征选择又能处理共线性。
            
- `sklearn.tree.DecisionTreeRegressor()`: **决策树回归**。
    
    - **特点：** 能处理非线性关系，可解释性较好。
        
    - **缺点：** 容易过拟合，对数据波动敏感。
        
- `sklearn.ensemble.RandomForestRegressor()`: **随机森林回归**。
    
    - **特点：** 集成学习，多棵决策树的平均结果，鲁棒性强，不易过拟合，能处理高维数据和非线性。
        
    - **缺点：** 模型解释性不如单棵决策树直观。
        
- `sklearn.svm.SVR()`: **支持向量回归**。
    
    - **特点：** 通过核函数处理非线性，对异常值不敏感（通过容忍带）。
        
    - **缺点：** 大数据集上计算量大，参数调优复杂。
        

#### 4. 聚类模型 (Clustering Models) - **重点**

用于发现数据中内在结构或分组（无监督学习）。

- `sklearn.cluster`:
    
    - `KMeans()`: **K均值聚类**。
        
        - **原理：** 将数据点分配到最近的K个质心，并迭代更新质心位置。
            
        - **特点：** 简单、高效、易于理解。
            
        - **缺点：** 需要预设K值，对初始质心和异常值敏感，只能发现球形簇。
            
        - `kmeans = KMeans(n_clusters=3, random_state=0)`
            
        - `kmeans.fit(X)`
            
        - `kmeans.labels_` (获取每个点的簇标签)
            
    - `DBSCAN()`: **基于密度的空间聚类**。
        
        - **原理：** 根据数据点的密度连接性来发现任意形状的簇，能识别噪声点。
            
        - **特点：** 不需要预设K值，能发现任意形状的簇，对噪声鲁棒。
            
        - **缺点：** 对参数 `eps` (邻域半径) 和 `min_samples` (最小样本数) 敏感，不适用于密度差异大的数据集。
            
        - `dbscan = DBSCAN(eps=0.5, min_samples=5)`
            
        - `dbscan.fit(X)`
            
        - `dbscan.labels_`
            
    - `AgglomerativeClustering()`: **层次聚类（凝聚型）**。
        
        - **原理：** 从每个数据点为一个簇开始，逐步合并最近的簇，直到达到指定数量的簇或满足停止条件。
            
        - **特点：** 不需要预设K值（但可以指定），可以生成树状图（dendrogram）展示聚类过程，能发现不同尺度的簇。
            
        - **缺点：** 计算复杂度高（尤其在大数据集上），对噪声和异常值敏感，一旦合并无法撤销。
            
        - `agg_clustering = AgglomerativeClustering(n_clusters=3)`
            
        - `agg_clustering.fit(X)`
            
        - `agg_clustering.labels_`
            

#### 5. 分类模型 (Classification Models) - **常用**

用于预测离散类别的模型。

- `sklearn.linear_model.LogisticRegression()`: **逻辑回归**。
    
    - **特点：** 线性分类器，输出概率，简单高效。
        
- `sklearn.svm.SVC()`: **支持向量分类**。
    
    - **特点：** 通过核函数处理非线性，寻找最优超平面。
        
- `sklearn.tree.DecisionTreeClassifier()`: **决策树分类**。
    
    - **特点：** 树状结构，易于理解。
        
- `sklearn.ensemble.RandomForestClassifier()`: **随机森林分类**。
    
    - **特点：** 集成多棵决策树，鲁棒性好，不易过拟合。
        

#### 6. 模型评估 (Metrics)

用于衡量模型性能。

- `sklearn.metrics`:
    
    - **回归：** `mean_squared_error`, `mean_absolute_error`, `r2_score`。
        
    - **分类：** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`。
        
    - **聚类：** `silhouette_score` (轮廓系数，衡量聚类效果好坏)。
        

### 三、总结

`sklearn` 是一个功能强大且易于使用的机器学习库，通过其模块化的设计，可以高效地完成从数据预处理到模型训练、评估和调优的整个机器学习流程。熟练掌握其核心模块和函数是进行数据科学项目的基础。