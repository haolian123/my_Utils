#数学建模常用方法类
import numpy as np
import math
from HaoChiUtils import DataAnalyzer,DataPreprocess
# @by haolian
class MathModeling:
    def __init__(self) -> None:
        pass


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
    ############################最小二乘法示例################################
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
        # x = np.array([1, 2, 3, 4, 5])
        # y = np.array([1, 3, 4, 3, 8])

        # MathModeling.least_squares(x,y)
    ##########################################################################



    
if __name__ =='__main__':

    pass 