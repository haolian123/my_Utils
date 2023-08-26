#数学建模常用方法类
import numpy as np
import sys
sys.path.append(rf'D:\学习资料\HCYNLP\MachineLearning')
import math
from HaoChiUtils import DataAnalyzer,DataPreprocess
import scipy.stats as stats
import heapq
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif']=['SimHei']
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# @by haolian 2023年8月
class MathModeling:
    def __init__(self) -> None:
        pass

    #拉格朗日插值法填补空值
   # 使用拉格朗日插值法填补缺失值
    @classmethod
    def lagrange_interpolation(self,x_known, y_known, x_missing):
        weights = []
        for i in range(len(x_known)):
            weight = 1
            for j in range(len(x_known)):
                if i != j:
                    weight *= (x_missing - x_known[j]) / (x_known[i] - x_known[j])
            weights.append(weight)
        y_missing = np.sum(weights * y_known)
        return y_missing

    #牛顿插值法
    @classmethod
    def newton_interpolation(x_known, y_known, x_missing):
        n = len(x_known)
        coefficients = [y_known[0]]
        for i in range(1, n):
            divided_differences = []
            for j in range(i, n):
                divided_difference = (y_known[j] - y_known[j-1]) / (x_known[j] - x_known[j-i])
                divided_differences.append(divided_difference)
            coefficients.append(divided_differences[0])
            for k in range(1, i+1):
                coefficients[i] *= (x_missing - x_known[k-1])
        y_missing = sum(coefficients)
        return y_missing
    ##############################使用例子#############################
    # 已知数据点
    # x_known = np.array([1, 2, 4, 5])
    # y_known = np.array([3, 5, 6, 8])
    # # 需要填补的自变量值
    # x_missing = 3
    # # 填补缺失值
    # y_missing = lagrange_interpolation(x_known, y_known, x_missing)
    # print("缺失值的填补结果为:", y_missing)
    ###################################################################

    #一致性检验
    # 如果满足一致性检验则返回权重向量，否则返回None
    @classmethod
    def consistency_check(self, matrix):
        # 检查矩阵的一致性
        n = matrix.shape[0]
        if n < 2:
            print("矩阵太小，无法进行一致性检验！")
            return
        
        # 计算矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_eigenvalue = max(eigenvalues.real)
        index = np.where(eigenvalues.real == max_eigenvalue)[0][0]
        eigenvector = eigenvectors[:, index].real
        normalized_eigenvector = eigenvector / np.sum(eigenvector)

        # 计算一致性指标CI
        lambda_max = max_eigenvalue
        random_index = np.array([0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51,1.54,1.56,1.58,1.59,1.5943])
        consistency_index = (lambda_max - n) / (n - 1)
        consistency_ratio = consistency_index / random_index[n - 1]
        
        # 判断一致性是否通过
        if consistency_ratio < 0.1:
            return normalized_eigenvector
        else:
            return None

        ##################################一致性检验示例############################
        # matrix = np.array([[1, 3, 5],
        #                 [1/3, 1, 2],
        #                 [1/5, 1/2, 1]])
        #
        # normalized_eigenvector=consistency_check(matrix)
        ##########################################################################


    # Topsis法
    # 对指标进行对象的评分排名，将所有的指标转化为极大型指标
    # 层次分析法的局限性：决策层不能过多，利用决策层指标已知数据
    # impacts[i]有四种类型 max 、min 、interval 、mid
    # 数值越接近1，表示该数据点越接近理想解；数值越接近0，表示该数据点越接近负理想解。
    @classmethod
    def topsis(self, data, weights, impacts):
        # 正向化
        normalized_data = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            if impacts[i] == 'min':
                normalized_data[:, i] = self.__topsis_min(datas=data[:, i], x_max=np.max(data[:, i]))
            elif impacts[i] == 'mid':
                normalized_data[:, i] = self.__topsis_mid(data[:, i], np.min(data[:, i]), np.max(data[:, i]))
            elif impacts[i] == 'interval':
                normalized_data[:, i] = self.__topsis_interval(data[:, i], np.min(data[:, i]), np.max(data[:, i]), np.min(data[:, i]), np.max(data[:, i]))
            elif impacts[i] == 'max':
                normalized_data[:, i] = data[:, i]
            else:
                raise Exception("类型不正确！")

        # 标准化
        for i in range(data.shape[1]):
            normalized_data[:, i] = normalized_data[:, i] / math.sqrt((sum(normalized_data[:, i] ** 2)))

        # 计算综合评价指标
        scores = np.zeros(data.shape[0], dtype=float)
        for i in range(data.shape[0]):
            scores[i] = sum(normalized_data[i, :] * weights)

        return scores
    @classmethod
    def __topsis_min(self,datas, x_max):
        def normalization(data):
            return x_max-data

        return list(map(normalization, datas))

    @classmethod
    def __topsis_mid(self,datas, x_min, x_max):
        def normalization(data):
            if data <= x_min or data >= x_max:
                return 0
            elif data > x_min and data < (x_min + x_max) / 2:
                return 2 * (data - x_min) / (x_max - x_min)
            elif data < x_max and data >= (x_min + x_max) / 2:
                return 2 * (x_max - data) / (x_max - x_min)

        return list(map(normalization, datas))

    @classmethod
    def __topsis_interval(self,datas, x_min, x_max, x_minimum, x_maximum):
        def normalization(data):
            if data >= x_min and data <= x_max:
                return 1
            elif data <= x_minimum or data >= x_maximum:
                return 0
            elif data > x_max and data < x_maximum:
                return 1 - (data - x_max) / (x_maximum - x_max)
            elif data < x_min and data > x_minimum:
                return 1 - (x_min - data) / (x_min - x_minimum)

        return list(map(normalization, datas))
    
        ##################################topsis示例############################
        # data = np.array([
        #     [1000, 2000, 3000],
        #     [150, 250, 350],
        #     [20, 30, 40],
        #     [2, 3, 4]
        # ])

        # weights = np.array([0.4, 0.3, 0.3])
        # impacts = ["max",'max','max']
        # res=MathModeling.topsis(data,weights,impacts)
        # print(res)
        ##########################################################################
    
    #拟合算法(最小二乘法)
    #返回斜率k 和 截距 b
    # 线性函数才可以用，否则用sse(误差和,求平均后即均方误差mse)
    
    @classmethod
    def least_squares(self,X,Y):
        # 使用最小二乘法拟合直线 y = kx + b
        A = np.vstack([X, np.ones(len(X))]).T
        k,b = np.linalg.lstsq(A, Y, rcond=None)[0]
        #计算决定系数R2
        predicted_values=[k*x+b for  x in X]
        R2=DataAnalyzer.get_R2(actual_values=Y,predicted_values=predicted_values)
        print("决定系数R2=",R2)
        return k,b
    ############################最小二乘法示例################################
        # x = np.array([1, 2, 3, 4, 5])
        # y = np.array([1, 3, 4, 3, 8])

        # MathModeling.least_squares(x,y)
    ##########################################################################

    #相关系数和假设性检验
    #返回皮尔逊相关系数,p为显著性水平
    @classmethod 
    def pearsonr(self,X,Y,p=0.05):
        pearson_coef, p_value = stats.pearsonr(X, Y)

        # 进行假设性检验
        alpha = p  # 设置显著性水平

        if p_value < alpha:
            print("拒绝原假设，两个变量间存在线性关系")
        else:
            print("接受原假设，两个变量间不存在线性关系")
        return pearson_coef
    #########################皮尔逊相关系数示例############################
    # 样本数据
    # x = [2, 4, 6, 8, 10]  # 学习时间
    # y = [60, 70, 80, 50, 10]  # 考试分数

    # # 计算皮尔逊系数
    # pearson_coef, p_value = stats.pearsonr(x, y)
    ######################################################################

    #得到相关系数矩阵
    #例如data = np.array([[170, 165, 180], [65, 60, 70], [30, 25, 35]])
    @classmethod
    def correlation_matrix(self,data):
        # 计算相关系数矩阵
        correlation_matrix = np.corrcoef(data)
        return correlation_matrix
    ###########################相关系数矩阵示例#############################
    # data = np.array([[170, 165, 180], [65, 60, 70], [30, 25, 35]])
    # # 计算相关系数矩阵
    # correlation_matrix = np.corrcoef(data)
    #######################################################################

    


    #图论
    #Floyd算法
    @classmethod
    def floyd(self,graph):
        n = len(graph)
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0

        for u in range(n):
            for v in range(n):
                dist[u][v] = graph[u][v]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist
    
    #Dijkstra算法
    @classmethod
    def dijkstra(self,graph,start=0):
        n = len(graph)
        dist = [float('inf')] * n
        dist[start] = 0
        visited = [False] * n
        heap = [(0, start)]

        while heap:
            cost, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True

            for v in range(n):
                if graph[u][v] != 0 and dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]
                    heapq.heappush(heap, (dist[v], v))

        return dist
    ################################图的算法示例###################################
    #  # 有向图的邻接矩阵表示
    #     graph = [
    #         [0, 3, 8, float('inf'), 4],
    #         [float('inf'), 0, float('inf'), 1, 7],
    #         [float('inf'), 4, 0, float('inf'), float('inf')],
    #         [2, float('inf'), 5, 0, float('inf')],
    #         [float('inf'), float('inf'), float('inf'), 6, 0]
    #     ]
    #     # 使用Floyd算法计算所有节点之间的最短路径
    #     floyd_result = MathModeling.floyd(graph)
    #     print("Floyd算法结果：")
    #     for row in floyd_result:
    #         print(row)
    #     # 使用Dijkstra算法计算从节点0到其他节点的最短路径
    #     dijkstra_result = MathModeling.dijkstra(graph, 0)
    #     print("Dijkstra算法结果：")
    #     print(dijkstra_result)
    ############################################################################


    #  聚类算法
    #  输入向量的列表，返回标签


    #K-Means聚类算法
    @classmethod
    def kmeans(self,data,n_clusters=2):
        # 创建K-means模型
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)

        # 获取簇的中心点和标签
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # 绘制数据点和簇的中心点
        colors = ["g.", "r.", "b.", "c.", "m.", "y."]
        for i in range(len(data)):
            plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10)
        plt.show()
        #返回标签
        labels 
    ###############################kmeans例子#####################################
    # X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # # 定义要聚类的簇数
    # n_clusters = 2
    # # 调用封装的方法进行聚类和绘图
    # kmeans(X, n_clusters)
    ##############################################################################

    #层次聚类算法
    # threshold：距离值
    @classmethod
    def hierarchical(self,data,threshold):
        # 使用ward方法进行层次聚类
        linkage_matrix = linkage(data, method='ward')  

        # 绘制聚类树
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=range(len(data)), leaf_rotation=90)
        plt.xlabel('数据点')
        plt.ylabel('距离')
        plt.title('层次聚类树')

        # 利用threshold值对层次聚类结果进行切割，并返回簇标签
        labels = fcluster(linkage_matrix, threshold, criterion='distance')

        plt.show()
        #返回标签
        return labels
    

    #密度聚类算法
    # eps: 邻域半径。
    # min_samples: 最小样本数，用于确定核心点。
    @classmethod
    def DBSCAN(self,data,eps=0.3,min_samples=2):
         # 创建DBSCAN聚类器
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # 执行聚类
        labels = dbscan.fit_predict(data)

        #返回标签
        return labels
    
    #多元线性回归
    # 传入x y列表和待预测的new_x列表，返回结果列表
    @classmethod
    def linear_regression(self,x,y,new_x=[]):
        model=LinearRegression()
        model.fit(x,y)
        res_prediction=model.predict(new_x)
        return res_prediction
    #################################多元线性回归例子###################
        # X = np.array([[1, 2, 3],
        #       [4, 5, 6],
        #       [7, 8, 9]])

        # y = np.array([10, 20, 30])
        # new_data = np.array([[2, 3, 4],[5,6,7]])
        # # 进行多元线性回归分析
        # res=MathModeling.linear_regression(X, y,new_data)
        # print(res)
    ###################################################################


    #时间序列
    @classmethod
    #传入过去的时间和值、待预测的值
    def time_series(self,dates, values, future_dates):
        # 将日期转换为Pandas的datetime对象
        dates = pd.to_datetime(dates)
        future_dates = pd.to_datetime(future_dates)
        
        # 创建一个Pandas DataFrame
        df = pd.DataFrame({
            'dates': dates,
            'values': values
        })
        
        # 日期排序
        df.sort_values('dates', inplace=True)
        # 创建一个日期索引
        df.set_index('dates', inplace=True)
        # 将日期转换为数值形式，以便于模型训练
        X = (df.index - df.index[0]).days.values.reshape(-1, 1)
        y = df['values'].values
        # 使用简单的线性回归模型进行预测
        model = LinearRegression()
        model.fit(X, y)
        # 准备未来的日期数据
        X_future = (future_dates - df.index[0]).days.values.reshape(-1, 1)
        # 进行预测
        y_future = model.predict(X_future)
        # 创建一个新的DataFrame来存储预测结果
        future_df = pd.DataFrame({
            'dates': future_dates,
            'values': y_future
        })
        future_df.set_index('dates', inplace=True)
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['values'], label='Observed')
        plt.plot(future_df.index, future_df['values'], label='Predicted', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()
        plt.show()
        return future_df['values']
        #######################################时间序列例子############################################
        # dates = ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05']
        # values = [1, 2, 3,4,5]
        # future_dates = ['2021-01-06', '2021-01-07', '2021-01-08']
        # res=MathModeling.time_series(dates, values, future_dates)
        ##############################################################################################


if __name__ =='__main__':

    pass 