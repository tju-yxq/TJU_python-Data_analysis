1. 聚类是一种将数据集划分为多个“簇”的技术，其中同一簇内的元素相似度高，不同簇之间的元素相似度较低。是一种无监督学习算法，不需要预先标注数据标签

2. 聚类算法分为划分型聚类（K-Means）、层次聚类、基于密度的聚类（DBSCAN）和基于概率的聚类（高斯混合模型）

3. 聚类算法的性能度量分为内部指标（DB指数、Dunn指数）和外部指标（Jaccard系数、FM指数、Rand指数）

4. 聚类算法中的距离满足非负性（距离大于等于0）、同一性（距离为0时为本身到本身）、对称性（互换i,j距离值相等）和三角不等式（两边之和大于等于第三边）。

5. k-means++是k-means聚类算法中的一种初始化质心方法，可以减少聚类结果对初始化的敏感性，加快收敛速度，是通过从不断从当前质点附近选取最远较远的数据点作为新的质点的方法。

6. 确认簇数K的方法分为肘部法和轮廓系数法

7. scikit-learn库常见工具

   1. `LabelEncoder()`将每个类别映射成一个唯一的标签

      ```python
      from sklrearn.preprocessing import labelEncoder
      #初始化LabelEncoder()
      le=LabelEncoder()
      #映射
      le_data=le.fit_transform(data)
      ```

   2. `MinMaxScaler()`将数据归一化

      ```python
      from sklearn.preprocessing import MinMaxScaler
      #初始化MinMaxScaler(),特征范围设为0到1
      mms=MinMaxScaler(feature_range=(0,1))
      #映射
      mms_data=mms.fit_transform(data)
      ```

   3. `Kmeans()`

      ```python
      from sklearn.cluster import KMeans
      #创建聚类数为K的Kmeans对象
      kmeans=KMeans(n_clusters=k,random_state=0)
      #对数据进行聚类
      kmeans.fit(data)
      kmeans.cluster_centers_ #簇中心坐标
      kmeans.interia_ #误差平方和
      ```

      
