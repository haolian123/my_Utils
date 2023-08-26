# 介绍

这是一个用于数学建模的工具包，其中包含

1. 拉格朗日插值法
2. 牛顿插值法
3. 一致性检验

# 方法

## 插值法

### 拉格朗日插值法示例

```python
x_known = np.array([1, 2, 4, 5])
y_known = np.array([3, 5, 6, 8])
x_missing = 3
y_missing = MathModeling.lagrange_interpolation(x_known, y_known, x_missing)
print("缺失值的填补结果为:", y_missing)
```

### 牛顿插值法示例

```python
y_missing = MathModeling.newton_interpolation(x_known, y_known, x_missing)
```

## 一致性检验示例

```python
matrix = np.array([[1, 3, 5],
                  [1/3, 1, 2],
                  [1/5, 1/2, 1]])
normalized_eigenvector = MathModeling.consistency_check(matrix)
if normalized_eigenvector is not None:
    print("通过一致性检验，权重向量为:", normalized_eigenvector)
else:
    print("未通过一致性检验")
```

## Topsis法

Topsis 法是一种用于多指标决策分析的方法，它将多个指标进行对象的评分排名，并将这些指标转化为极大型指标。该方法主要用于帮助决策者从多个备选方案中选择最优方案。

本工具包提供了以下功能：

1. **Topsis 方法**

   使用 Topsis 法对给定的数据进行多指标决策分析。需要提供以下输入：

   - `data`：包含备选方案的数据矩阵，其中每一行代表一个备选方案，每一列代表一个评价指标。
   - `weights`：指标的权重，用于衡量各指标的重要性。
   - `impacts`：指标的影响类型，可以是 'max'、'min'、'interval' 或 'mid' 中的一种。

   ```python
   data = np.array([
       [1000, 2000, 3000],
       [150, 250, 350],
       [20, 30, 40],
       [2, 3, 4]
   ])

   weights = np.array([0.4, 0.3, 0.3])
   impacts = ["max", "max", "max"]

   # 运行 Topsis 方法
   res = MathModeling.topsis(data, weights, impacts)
   print(res)
   ```

2. **支持的影响类型**

   - `'max'`：指标越大越好。
   - `'min'`：指标越小越好。
   - `'interval'`：指标在一个区间内，区间两端的值越接近理想解越好。
   - `'mid'`：指标在一个区间内，区间中点的值越接近理想解越好。

## 拟合算法 (最小二乘法)
这段代码实现了最小二乘法用于线性拟合的功能。它计算出拟合的直线的斜率和截距，并计算决定系数 R2 以评估拟合的好坏。

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 4, 3, 8])
MathModeling.least_squares(x, y)
```

## 相关系数

### 相关系数和假设性检验

这部分代码计算了两个变量的皮尔逊相关系数，并根据显著性水平进行了假设性检验。

计算皮尔逊相关系数并进行假设性检验，其中 X 和 Y 分别是两个变量的数据，p 是显著性水平。

```python
x = [2, 4, 6, 8, 10]  # 学习时间
y = [60, 70, 80, 50, 10]  # 考试分数
pearson_coef = MathModeling.pearsonr(x, y)
```

### 得到相关系数矩阵
计算给定数据的相关系数矩阵，其中 data 是一个二维数组，每行代表一个变量，每列代表不同的观测。

```python
data = np.array([[170, 165, 180], [65, 60, 70], [30, 25, 35]])
correlation_matrix = MathModeling.correlation_matrix(data)
```

## 图论算法

### 最短路径

#### Floyd算法

Floyd算法用于计算图中任意两个节点之间的最短路径。

#### Dijkstra算法

Dijkstra算法用于计算图中某个节点到其他所有节点的最短路径,需要指定起点。

#### 图两个最短路径的算法示例

以下是一个有向图的邻接矩阵表示以及如何使用上述算法计算最短路径的示例代码：

```python
# 有向图的邻接矩阵表示
graph = [
    [0, 3, 8, float('inf'), 4],
    [float('inf'), 0, float('inf'), 1, 7],
    [float('inf'), 4, 0, float('inf'), float('inf')],
    [2, float('inf'), 5, 0, float('inf')],
    [float('inf'), float('inf'), float('inf'), 6, 0]
]

# 使用Floyd算法计算所有节点之间的最短路径
floyd_result = MathModeling.floyd(graph)
print("Floyd算法结果：")
for row in floyd_result:
    print(row)

# 使用Dijkstra算法计算从节点0到其他节点的最短路径
dijkstra_result = MathModeling.dijkstra(graph, 0)
print("Dijkstra算法结果：")
print(dijkstra_result)
```

## 聚类算法

### K-Means 聚类算法

K-Means 聚类是一种常用的基于距离的聚类算法，它将数据点分成预先指定数量的簇，以使每个数据点与其所属簇的中心点之间的距离最小化。

示例

```python
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
n_clusters = 2
labels = YourClass.kmeans(X, n_clusters)
```

### 层次聚类算法

层次聚类算法将数据点逐步合并到越来越大的簇中，形成一棵聚类树。它通过计算数据点之间的相似性来构建聚类树。

示例

```python
threshold = 5  # 根据实际情况调整阈值
labels = YourClass.hierarchical(X, threshold)
```

### 密度聚类算法

密度聚类算法将数据点分成高密度区域和低密度区域，不需要预先指定簇的数量。它适用于具有不规则形状和噪声的数据集。

示例

```python
eps = 0.5  # 根据实际情况调整邻域半径和最小样本数
min_samples = 3
labels = YourClass.DBSCAN(X, eps, min_samples)
```

## 多元线性回归

多元线性回归是一种用于建立多个自变量和一个因变量之间关系的统计分析方法。

### 示例

```python
# 示例数据
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
y = np.array([10, 20, 30])
new_data = np.array([[2, 3, 4], [5, 6, 7]])

# 进行多元线性回归分析
res = MathModeling.linear_regression(X, y, new_data)
print(res)
```

## 时间序列预测

### 输入参数

- dates: 过去的日期列表，格式为字符串，如['2021-01-01', '2021-01-02', '2021-01-03']
- values: 过去的数值列表，格式为数字，如[1, 2, 3]
- future_dates: 待预测的日期列表，格式为字符串，如['2021-01-04', '2021-01-05']

### 示例

```python
from MathModeling import MathModeling

dates = ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05']
values = [1, 2, 3, 4, 5]
future_dates = ['2021-01-06', '2021-01-07', '2021-01-08']

res = MathModeling.time_series(dates, values, future_dates)
print(res)
```

# 注意事项

1. 请确保安装了需要用到的库，在终端输入 pip install  所缺的库名

2. 在运行的文件中 import MathModeling, 即可按如下调用方法

   ```python
   import MathModeling 
   #使用方法
   MathModeling.方法(参数1,参数2)
   ```

3. ...