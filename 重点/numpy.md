# 一、Numpy

NumPy (Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

## 1.1 数组 (ndarray)

NumPy的核心数据结构是`ndarray`（N-dimensional array），即多维数组。

### 数组维度 (Dimensions)

* **轴 (axis)**: 数组的维度称为轴。例如，一个二维数组（矩阵）有2个轴：`axis=0` 代表沿着行的方向（通常指操作会应用到每一列），`axis=1` 代表沿着列的方向（通常指操作会应用到每一行）。
* **形状 (shape)**: 数组的形状是一个元组，表示数组在每个维度上的大小。
    ```python
    import numpy as np
    
    # 一维数组
    arr1d = np.array([1, 2, 3])
    print(f"一维数组: {arr1d}")
    print(f"形状: {arr1d.shape}") # 输出: (3,) 表示有3个元素
    
    # 二维数组
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"二维数组:\n{arr2d}")
    print(f"形状: {arr2d.shape}") # 输出: (2, 3) 表示2行3列
    
    # 三维数组
    arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"三维数组:\n{arr3d}")
    print(f"形状: {arr3d.shape}") # 输出: (2, 2, 2) 表示2个2x2的矩阵
    ```

* **改变形状**:
    * `reshape(new_shape)`: 返回一个具有新形状的数组，但**不改变**原始数组。新形状的元素总数必须与原形状一致。
        ```python
        a = np.arange(6) # array([0, 1, 2, 3, 4, 5])
        b = a.reshape((2, 3))
        print(f"reshape后的数组 b:\n{b}")
        # [[0 1 2]
        #  [3 4 5]]
        print(f"原始数组 a 仍然是: {a}") # a 未改变
        # [0 1 2 3 4 5]
        
        # np.reshape(array, new_shape) 效果相同
        c = np.reshape(a, (3, 2))
        print(f"np.reshape后的数组 c:\n{c}")
        # [[0 1]
        #  [2 3]
        #  [4 5]]
        ```
    * `resize(new_shape)`: **直接修改**原始数组的形状。如果新形状的元素总数与原形状不同：
        * 元素减少：如果新尺寸更小，数据会被舍弃 (需要 `refcheck=False` 来允许这种操作，否则可能报错)。
        * 元素增加：如果新尺寸更大，会用0来填充不足的元素。
        ```python
        a = np.arange(6)
        a.resize((3, 2)) # 直接修改 a
        print(f"resize后的数组 a:\n{a}")
        # [[0 1]
        #  [2 3]
        #  [4 5]]
        
        a.resize((3, 3)) # 元素增加，用0填充
        print(f"resize增加元素后的数组 a:\n{a}")
        # [[0 1 0]
        #  [2 3 0]
        #  [4 5 0]]
        
        # 注意：np.resize(array, new_shape) 的行为不同
        # 它返回一个新数组，如果新尺寸更大，会重复使用原数组的元素来填充
        original_array = np.array([1, 2, 3])
        new_array_np_resize = np.resize(original_array, (2, 3))
        print(f"np.resize 后的新数组:\n{new_array_np_resize}")
        # [[1 2 3]
        #  [1 2 3]]
        print(f"np.resize 后原始数组不变: {original_array}")
        # [1 2 3]
        ```

### 数组类型 (Data Types)

* `type(array)`: 查看数组对象的类型，通常是 `numpy.ndarray`。
* `array.dtype`: 查看数组中元素的数据类型。NumPy支持多种数据类型，如 `int32`, `int64`, `float32`, `float64`, `bool_`, `complex_`, `string_` 等。
* `array.astype(new_type)`: 转换数组元素的数据类型，返回一个新的数组。
    ```python
    arr = np.array([1, 2, 3])
    print(f"数组类型: {type(arr)}")    # <class 'numpy.ndarray'>
    print(f"元素数据类型: {arr.dtype}") # int64 (或 int32, 取决于系统)
    
    float_arr = arr.astype(np.float64)
    print(f"转换为float64后的数组: {float_arr}") # [1. 2. 3.]
    print(f"新数组的元素数据类型: {float_arr.dtype}") # float64
    
    numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
    float_from_strings = numeric_strings.astype(float) # 字符串转为浮点数
    print(f"字符串转浮点数: {float_from_strings}") # [  1.25  -9.6   42.  ]
    ```

### 数组创建

* `np.array(object, dtype=None, ...)`: 从列表、元组等序列型对象创建数组。
    ```python
    list_data = [1, 2, 3, 4, 5]
    arr_from_list = np.array(list_data)
    print(f"从列表创建的数组: {arr_from_list}")
    
    nested_list_data = [[1, 2], [3, 4]]
    arr_from_nested_list = np.array(nested_list_data)
    print(f"从嵌套列表创建的二维数组:\n{arr_from_nested_list}")
    ```
    
* `np.arange([start,] stop[, step,], dtype=None)`: 创建等差序列数组（类似于Python的`range`）。
    ```python
    arr_arange1 = np.arange(5) # 0 到 4
    print(f"np.arange(5): {arr_arange1}") # [0 1 2 3 4]
    
    arr_arange2 = np.arange(1, 10, 2) # 1 到 9，步长为 2
    print(f"np.arange(1, 10, 2): {arr_arange2}") # [1 3 5 7 9]
    ```
    
* 特殊数组创建：
    * `np.zeros(shape, dtype=float)`: 创建全0数组。
    * `np.ones(shape, dtype=float)`: 创建全1数组。
    * `np.eye(N, M=None, k=0, dtype=float)`: 创建单位矩阵 (N x M，对角线为1，其余为0)。`k`指定对角线位置。
    ```python
    zeros_arr = np.zeros((2, 3))
    print(f"全0数组:\n{zeros_arr}")
    
    ones_arr = np.ones((3, 2), dtype=int)
    print(f"全1整数数组:\n{ones_arr}")
    
    eye_arr = np.eye(3)
    print(f"3x3单位矩阵:\n{eye_arr}")
    ```

### 数组索引和切片 (Indexing and Slicing)

* **基本索引**: 与Python列表类似，从0开始。
    ```python
    arr = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]
    print(f"arr[0]: {arr[0]}")   # 0
    print(f"arr[5]: {arr[5]}")   # 5
    arr[5] = 100
    print(f"修改后 arr[5]: {arr[5]}") # 100
    ```
    
* **切片 (Slicing)**: `array[start:stop:step]`
    * 切片返回的是原始数组的**视图 (view)**，而不是副本。修改视图会影响原始数组。
    * 如果需要副本，使用 `.copy()` 方法。
    ```python
    arr = np.arange(10)
    slice_arr = arr[2:7:2] # 从索引2到6，步长为2
    print(f"切片 arr[2:7:2]: {slice_arr}") # [2 4 6]
    
    slice_arr[0] = 99 # 修改视图
    print(f"修改视图后，slice_arr: {slice_arr}") # [99  4  6]
    print(f"修改视图后，原始数组 arr: {arr}")   # [ 0  1 99  3  4  5  6  7  8  9] (原始数组也被修改)
    
    arr_copy_slice = arr[2:5].copy() # 创建副本
    arr_copy_slice[0] = 777
    print(f"修改副本后，arr_copy_slice: {arr_copy_slice}") # [777 3 4]
    print(f"修改副本后，原始数组 arr 未变: {arr}")        # [ 0  1 99  3  4  5  6  7  8  9]
    ```
    
* **`slice` 对象**: `slice(start, stop, step)` 可以创建一个切片对象，用于重复使用。
    ```python
    s = slice(1, 8, 2)
    print(f"使用slice对象 arr[s]: {arr[s]}") # [ 1 99  5]
    ```
    
* **高维数组索引和切片**:
    * 可以使用逗号分隔的索引元组 `arr[row, col]` 或 `arr[slice_row, slice_col]`。
    * 单个索引会降低维度。
    ```python
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"arr2d[1, 2]: {arr2d[1, 2]}") # 第2行第3列元素: 6
    
    # 切片：前两行，从第二列开始到末尾
    print(f"arr2d[:2, 1:]:\n{arr2d[:2, 1:]}")
    # [[2 3]
    #  [5 6]]
    
    # 获取第一行 (结果是一维数组)
    print(f"arr2d[0]: {arr2d[0]}") # [1 2 3]
    # 等同于 arr2d[0, :]
    
    # 获取第一列 (结果是一维数组)
    print(f"arr2d[:, 0]: {arr2d[:, 0]}") # [1 4 7]
    ```
    
* **布尔型索引 (Boolean Indexing)**:
    * 使用布尔数组作为索引，选择布尔数组中对应 `True` 位置的元素。
    * 布尔数组的长度必须与被索引的轴的长度一致。
    * 布尔索引总是创建数据的**副本**。
    ```python
    names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
    data = np.random.randn(4, 3) # 4x3的随机数据
    print(f"原始数据 data:\n{data}")
    
    # 选取名字是 'Bob' 的所有行
    bob_rows = data[names == 'Bob'] # names == 'Bob' 产生布尔数组 [True, False, False, True]
    print(f"名字是 'Bob' 的行:\n{bob_rows}")
    
    # 也可以组合条件
    # 选取名字是 'Bob' 或 'Will' 的行
    mask = (names == 'Bob') | (names == 'Will') # 使用 | (或), & (与), ~ (非)
    print(f"名字是 'Bob' 或 'Will' 的行:\n{data[mask]}")
    
    # 对小于0的元素赋值为0
    data_copy = data.copy()
    data_copy[data_copy < 0] = 0
    print(f"将负数置0后的 data_copy:\n{data_copy}")
    ```
    
* **`np.where(condition[, x, y])`**:
    * 如果只有 `condition`：返回满足条件的元素的索引元组。
    * 如果提供 `x` 和 `y`：根据 `condition` 从 `x` (True) 或 `y` (False) 中选取元素。
    ```python
    arr = np.random.randn(3, 3)
    print(f"随机数组 arr:\n{arr}")
    # 将正数替换为1，负数替换为-1，0保持为0 (np.sign也可)
    result_where = np.where(arr > 0, 1, np.where(arr < 0, -1, 0))
    print(f"np.where 处理后的数组:\n{result_where}")
    
    # 找出大于0.5的元素的索引
    indices = np.where(arr > 0.5)
    print(f"大于0.5的元素的索引 (行, 列):\n{indices}")
    print(f"这些元素是: {arr[indices]}")
    ```

### 字符串操作 (`np.char`)

NumPy提供了一系列用于对字符串数组进行矢量化操作的函数，位于 `np.char` 模块。

* `np.char.add(arr1, arr2)`: 字符串连接

* `np.char.multiply(arr, num)`: 字符串重复。

* `np.char.split(arr, sep=None)`: 字符串分割。

* `np.char.lower(arr)`, `np.char.upper(arr)`, `np.char.capitalize(arr)`, `np.char.title(arr)`: 大小写转换。

* `np.char.strip(arr, chars=None)`: 去除首尾字符。

* `np.char.join(sep, arr)`: 用分隔符连接数组中的字符串序列。

* `np.char.replace(arr, old, new, count=None)`: 替换子字符串。

* 比较函数如 `np.char.equal()`, `np.char.not_equal()`, 等。
    ```python
    s1 = np.array(['hello', 'world'])
    s2 = np.array([' python', '!'])
    print(f"np.char.add: {np.char.add(s1, s2)}") # ['hello python' 'world!']
    
    print(f"np.char.upper: {np.char.upper(s1)}") # ['HELLO' 'WORLD']
    
    names = np.array([' apple ', ' banana ', ' cherry '])
    print(f"np.char.strip: {np.char.strip(names)}") # ['apple' 'banana' 'cherry']
    ```

### 数组排序

* `np.sort(a, axis=-1, kind=None, order=None)`: 返回数组 `a` 的排序**副本**。
    * `axis=0`: 沿列排序（对每一列的元素独立排序）。
    * `axis=1`: 沿行排序（对每一行的元素独立排序）。
* `np.argsort(a, axis=-1, kind=None, order=None)`: 返回排序后元素在原数组中的**索引**。
    ```python
    arr = np.array([[3, 1, 2], [6, 0, 4]])
    print(f"原始数组 arr:\n{arr}")
    
    sorted_arr_rows = np.sort(arr, axis=1) # 按行排序
    print(f"按行排序 (np.sort):\n{sorted_arr_rows}")
    # [[1 2 3]
    #  [0 4 6]]
    print(f"原始数组 arr 仍然是:\n{arr}") # np.sort 不改变原数组
    
    # argsort
    unsorted_array = np.array([50, 10, 30, 20, 40])
    indices = np.argsort(unsorted_array)
    print(f"argsort 得到的索引: {indices}") # [1 3 2 4 0]
    print(f"使用索引排序后的数组: {unsorted_array[indices]}") # [10 20 30 40 50]
    ```

### 数组合并与分割

#### 合并 (Concatenation / Stacking)

* `np.concatenate((a1, a2, ...), axis=0)`: 沿指定轴连接一系列数组。
    ```python
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    # 沿 axis=0 (行方向) 合并
    cat_axis0 = np.concatenate((a, b), axis=0)
    print(f"沿 axis=0 合并:\n{cat_axis0}")
    # [[1 2]
    #  [3 4]
    #  [5 6]]
    
    c = np.array([[7], [8]])
    # 沿 axis=1 (列方向) 合并 (a需要调整形状或c需要调整形状才能匹配)
    # 假设 a 和 c 的行数相同
    a_for_cat1 = np.array([[1,2],[3,4]])
    c_for_cat1 = np.array([[5],[6]])
    cat_axis1 = np.concatenate((a_for_cat1, c_for_cat1), axis=1)
    print(f"沿 axis=1 合并:\n{cat_axis1}")
    # [[1 2 5]
    #  [3 4 6]]
    ```
* `np.vstack((tup))` 或 `np.row_stack((tup))`: 垂直堆叠（沿行）。
* `np.hstack((tup))` 或 `np.column_stack((tup))`: 水平堆叠（沿列）。`column_stack` 对一维数组的处理更友好，会将其视为列向量。
    ```python
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    print(f"np.vstack:\n{np.vstack((arr1, arr2))}")
    # [[1 2 3]
    #  [4 5 6]]
    print(f"np.hstack:\n{np.hstack((arr1, arr2))}")
    # [1 2 3 4 5 6]
    
    arr2d_1 = np.array([[1],[2]])
    arr2d_2 = np.array([[3],[4]])
    print(f"np.hstack for 2D:\n{np.hstack((arr2d_1, arr2d_2))}")
    # [[1 3]
    #  [2 4]]
    ```
* `np.append(arr, values, axis=None)`: 将 `values` 追加到 `arr` 的末尾。如果指定 `axis`，行为类似于 `concatenate`。

#### 分割 (Splitting)

* `np.split(ary, indices_or_sections, axis=0)`: 将数组沿指定轴分割成多个子数组。
* `np.array_split(ary, indices_or_sections, axis=0)`: 与 `split` 类似，但如果不能等分，它不会报错，而是创建大小不等的子数组。
* `np.hsplit(ary, indices_or_sections)`: 水平分割（等价于 `split` 时 `axis=1`）。
* `np.vsplit(ary, indices_or_sections)`: 垂直分割（等价于 `split` 时 `axis=0`）。
    ```python
    arr = np.arange(12).reshape((3, 4))
    print(f"要分割的数组 arr:\n{arr}")
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]]
    
    # 垂直分割成3部分 (沿axis=0)
    v_splits = np.vsplit(arr, 3)
    print(f"垂直分割结果: 第1部分\n{v_splits[0]}") # [[0 1 2 3]]
    
    # 水平分割成2部分 (沿axis=1)
    h_splits = np.hsplit(arr, 2)
    print(f"水平分割结果: 第1部分\n{h_splits[0]}")
    # [[ 0  1]
    #  [ 4  5]
    #  [ 8  9]]
    ```

### 数组删除

* `np.delete(arr, obj, axis=None)`: 返回一个新的数组，其中删除了沿指定轴 `axis` 的子数组或元素 `obj`。
    * `obj` 可以是整数索引、整数索引列表或切片。
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始数组 arr:\n{arr}")
    
    # 删除第一行 (axis=0, obj=0)
    deleted_row = np.delete(arr, 0, axis=0)
    print(f"删除第一行后:\n{deleted_row}")
    # [[4 5 6]
    #  [7 8 9]]
    
    # 删除第二列 (axis=1, obj=1)
    deleted_col = np.delete(arr, 1, axis=1)
    print(f"删除第二列后:\n{deleted_col}")
    # [[1 3]
    #  [4 6]
    #  [7 9]]
    
    # 删除多个索引
    deleted_multi_rows = np.delete(arr, [0, 2], axis=0) # 删除第0行和第2行
    print(f"删除第0和2行后:\n{deleted_multi_rows}")
    # [[4 5 6]]
    ```

## 1.2 线性代数 (`numpy.linalg`)

NumPy 通过 `numpy.linalg` 模块提供了丰富的线性代数运算功能。

* **矩阵和向量积**:
    * `np.matmul(arr1,arr2)`、`np.dot(arr1,arr2)`、`A@B`实现二维矩阵乘法（行乘列）、一维数组点积
    * `arr1*arr2`、`np.vdot(arr1,arr2)`实现内积（矩阵元素对应相乘）
    * `np.cross(arr1,arr2)`实现叉积
    ```python
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print(f"v1 点积 v2 (np.dot): {np.dot(v1, v2)}") # 1*4 + 2*5 + 3*6 = 32
    print(f"v1 点积 v2 (@): {v1 @ v2}")           # 32
    
    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[5, 6], [7, 8]])
    print(f"矩阵 m1:\n{m1}")
    print(f"矩阵 m2:\n{m2}")
    print(f"m1 矩阵乘 m2 (np.matmul):\n{np.matmul(m1, m2)}")
    # [[1*5+2*7  1*6+2*8]  = [[19 22]
    #  [3*5+4*7  3*6+4*8]]    [43 50]]
    print(f"m1 矩阵乘 m2 (@):\n{m1 @ m2}")
    print(f"m1 元素级乘 m2 (*):\n{m1 * m2}")
    # [[ 5 12]
    #  [21 32]]
    ```
* **矩阵转置**: `array.T` 
    
    ```python
    m = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"矩阵 m:\n{m}")
    print(f"矩阵 m 的转置 (m.T):\n{m.T}")
    # [[1 4]
    #  [2 5]
    #  [3 6]]
    ```
* **行列式**: `np.linalg.det(a)` 计算方阵 `a` 的行列式。
    ```python
    m = np.array([[1, 2], [3, 4]])
    print(f"矩阵 m 的行列式: {np.linalg.det(m):.2f}") # 1*4 - 2*3 = -2.00
    ```
* **逆矩阵**: `np.linalg.inv(a)` 计算方阵 `a` 的逆矩阵 $A^{-1}$。
    ```python
    m = np.array([[1., 2.], [3., 4.]]) # 使用浮点数以避免整数运算问题
    m_inv = np.linalg.inv(m)
    print(f"矩阵 m 的逆矩阵:\n{m_inv}")
    # [[-2.   1. ]
    #  [ 1.5 -0.5]]
    # 验证: m @ m_inv 应该接近单位矩阵
    print(f"m @ m_inv (应接近单位矩阵):\n{np.round(m @ m_inv)}")
    ```
* **特征值和特征向量**: `np.linalg.eig(a)` 计算方阵 `a` 的特征值和右特征向量。
    * 返回一个元组 `(eigenvalues, eigenvectors)`。
    * `eigenvectors` 的每一列是一个对应于 `eigenvalues` 中相应特征值的特征向量。
    ```python
    m = np.array([[4, 2], [1, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(m)
    print(f"特征值: {eigenvalues}")
    print(f"特征向量 (每列一个):\n{eigenvectors}")
    # 验证: m @ v = lambda * v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_val = eigenvalues[i]
        print(f"验证特征向量 {i+1}: m @ v = {m @ v}, lambda * v = {lambda_val * v}")
        # 两者应该非常接近
    ```
* **解线性方程组**: `np.linalg.solve(a, b)` 解线性方程组 $Ax = b$，其中 `a` 是系数矩阵，`b` 是一维或二维数组（常数项）。
    ```python
    # 方程组:
    # x + 2y = 5
    # 3x + 4y = 11
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 11])
    x = np.linalg.solve(A, b)
    print(f"线性方程组的解 x (x, y): {x}") # [1. 2.] (即 x=1, y=2)
    # 验证: A @ x 应该等于 b
    print(f"验证: A @ x = {A @ x}") # [ 5. 11.]
    ```
