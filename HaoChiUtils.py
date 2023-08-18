# 导入所需的库
import re
import csv
import jieba
import emoji
from opencc import OpenCC
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import math
from sklearn.metrics import r2_score,roc_curve,auc
#================================================by Chi================================================
#定义数据预处理类 
#包含数据预处理的相关方法
class DataPreprocess:

    # 指定的停用词
    __stop_terms = ["展开", "全文", "转发", "显示原图", "原图","显示地图",'转发微博','分享图片']

    #停用词表
    __stopwords = []

    def __init__(self,stopwords_file_path = "hit_stopwords.txt") :
        # 加载停用词列表
        
        with open(stopwords_file_path, "r", encoding="utf-8") as stopwords_file:
            for line in stopwords_file:
                self.__stopwords.append(line.strip())

    
    # 定义清洗文本的函数
    def text_clean(self,text,has_user_id=False, keep_segmentation=False):
    #当keep_segmentation为False时，text_clean方法会使用jieba库对清洗后的文本进行分词处理，并返回分词后的结果。       

        # 使用OpenCC库将繁体中文转换为简体中文
        cc = OpenCC('t2s')
        text = cc.convert(text)

        #如果有用户id
        if has_user_id:
            # 去除冒号后的内容
            for i in range(len(text)):
                if text[i] == ':' or text[i] == '：':
                    text = text[i + 1:-1]
                    break

        # 定义中文标点符号和URL正则表达式
        zh_puncts1 = "，；、。！？（）《》【】\"\'"
        URL_REGEX = re.compile(
            r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' +
            zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
            r'[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',re.IGNORECASE)
        
        # 去除URL
        text = re.sub(URL_REGEX, "", text)
        # 去除@用户和回复标记
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)", " ", text)
        # 将表情符号转换为文本描述
        text = emoji.demojize(text)
        # 去除表情符号
        text = re.sub(r"\[\S+?\]", "", text)
        # 去除话题标签
        text = re.sub(r"#\S+#", "", text)
        # 去除数字
        text = re.sub(r'\d+', '', text)
         #去除中文标点
        # 使用re.sub()函数将标点符号替换为空格
        text = re.sub(r'[^\w\s]', ' ', text)
        # 去除多余的空格
        text = re.sub(r"(\s)+", r"\1", text)
        for x in self.__stop_terms:
            text = text.replace(x, "")
        # 去除首尾空格
        text = text.strip()
        if keep_segmentation:
            return text
        else:
            # 使用结巴分词进行分词
            seg_list = list(jieba.cut(text,cut_all=False))        
            # 去除停用词
            seg_list = [word for word in seg_list if word not in self.__stopwords]
            # 将分词结果拼接为字符串
            cleaned_text = ' '.join(seg_list)
        
        return cleaned_text


    #只能处理文件格式为 text label且以\t为分隔符 的文件
    def text_process(self,input_file_path="DataSet.tsv", output_file_path="Clean_data.tsv"):

        count=1
        # 打开输入文件并读取内容
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
            cleaned_lines = []
            
            # 遍历每一行数据
            for line in lines:
                
                line = line.strip().split('\t')
                if line[1]=="label":
                    print(line[1])
                    continue
                if count%500==0:
                    print(f"已处理{count}条文本记录")
                count+=1

                # 检查列表长度是否足够
                if len(line) == 2:
                    # 调用clean_text函数清洗第一列的文本数据，并保留其他几列数据
                    clean_text=self.text_clean(line[0])
                    # 删去第一列内容为空的行
                    if clean_text !='':
                        cleaned_line = [self.text_clean(line[0]),line[1]]

                    cleaned_lines.append(cleaned_line)
            
            # 打开输出文件并写入清洗后的数据，写入csv
            with open(output_file_path, "w", encoding="utf-8", newline='') as output_file:
                writer = csv.writer(output_file,delimiter='\t')
                for line in cleaned_lines:
                    writer.writerow(line)
            print(f"共有{len(cleaned_lines)}条记录！")
            # # 输出提示信息
            # print("修改后的内容已写入新文件。")

    #归一化
    @classmethod
    def normalization(self,data):
        # 创建MinMaxScaler对象
        scaler=MinMaxScaler()
        # 将数据集进行归一化处理
        normalized_data=scaler.fit_transform(data)
        return normalized_data
    
    #标准化
    @classmethod
    def standardization(self,data):
        # 创建StandardScaler对象
        scaler = StandardScaler()
        # 将数据集进行标准化处理
        standardized_data = scaler.fit_transform(data)
        return standardized_data


    #主成分分析和特征降维
    # 选择累计解释方差比例超过threshold(如95%)的主成分数量作为保留的主成分数量。
    @classmethod
    def pca(self,data,threshold=0.95):
        #创建PCA对象
        my_pca=PCA()

        #对数据进行主成分分析
        my_pca.fit(data)

        # 获取每个主成分的方差解释比例
        explained_variance_ratio = my_pca.explained_variance_ratio_

        # 计算累计解释方差比例
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # 找到累计解释方差比例超过阈值的主成分数量
        n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

        my_pca=PCA(n_components=n_components)

        X_pca=my_pca.fit_transform(data)

        return X_pca

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



# ================================by Hao=====================================
#数据分析类
#包含数据分析的相关方法
class DataAnalyzer:

    def __init__(self) -> None:
        pass

    #接受tsv文件
    #划分数据集为测试集、验证集、训练集
    @classmethod
    def split_dataSet(self,dataSet_path='dataSet.tsv'):
        # 读取数据集
        data = pd.read_csv(dataSet_path, delimiter='\t')

        # 划分训练集和剩余数据
        train_data, remaining_data = train_test_split(data, test_size=0.2, random_state=42)

        # 划分验证集和测试集
        valid_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

        # 保存划分后的数据集
        train_data.to_csv('train.tsv', sep='\t', index=False)
        valid_data.to_csv('eval.tsv', sep='\t', index=False)
        test_data.to_csv('test.tsv', sep='\t', index=False)

    # 绘制训练过程的曲线
    @classmethod
    def draw_process(self, title='training acc', color='r', iters=[], data=[], label='training acc', png_path='plot'):
        # 设置图表标题和字体大小
        plt.title(title, fontsize=24)
        # 设置x轴标签和字体大小
        plt.xlabel("iter", fontsize=20)
        # 设置y轴标签和字体大小
        plt.ylabel(label, fontsize=20)
        # 绘制曲线，使用指定的颜色和标签
        plt.plot(iters, data, color=color, label=label)
        # 添加图例
        plt.legend()
        # 添加网格线
        plt.grid()
        # 保存图表为PNG格式图片
        plt.savefig(png_path+'/'+label+'.png')
        # 显示图表（可选）
        # plt.show()


    # 计算标签占比，输入为预测结果的列表，输出为标签:占比(0.xx)
    @classmethod
    def calculate_label_proportions(self,predictions,label_list):
        # 创建一个空字典用于存储标签及其对应的数量
        predictions_dict = {}
        #添加字典
        for i in label_list:
            predictions_dict[i]=0
        # 初始化总数量为0
        total_cnt = 0
        # 遍历预测结果列表
        for prediction in predictions:
            # 如果标签不在字典的键中，则将其添加到字典并初始化数量为0
            if prediction not in predictions_dict.keys():
                predictions_dict[prediction] = 0
            # 将标签对应的数量加1
            predictions_dict[prediction] += 1
            # 总数量加1
            total_cnt += 1
        # # 创建一个空字典用于存储标签及其对应的占比
        res_dict = {}
        # 遍历标签及其对应的数量
        for key, value in predictions_dict.items():
            # # 计算标签的占比，并保留一位小数
            proportion=0
            if total_cnt!=0:
                proportion = round(value / total_cnt , 2)
            else:
                proportion=0
            # 将占比存储到子字典中
            res_dict[key] = proportion
        # 返回标签及其对应的占比字典
        return res_dict
    

    #转化为列表可供模型读取
    #data_file_path:    文件路径
    #min_len:           文本的最小长度
    @classmethod
    def get_dataList(self,data_file_path,min_len=1):
        ret_list=[]
        with open(data_file_path,'r',encoding='utf-8') as f:
            for data_line in f:
                data_line=data_line.strip().strip('\n')
                ##文本清洗：删除@、url、标点等无关信息
                #需要嵌入代码
                data_line=DataPreprocess.text_clean(text=data_line,has_user_id=False,keep_segmentation=True)
                #筛选大于等于规定长度的文本
                if data_line is not None and len(data_line) >=min_len:
                    ret_list.append(data_line)
        return ret_list


    #计算准确率
    #传入真实和预测列表
    #返回准确率，保留4位小数
    @classmethod
    def get_score(self,truth_label,predict_label):
        assert len(truth_label)==len(predict_label),'列表长度不一致！'
        cnt=0
        for i in range(len(truth_label)):
            if truth_label[i]==predict_label[i]:
                cnt+=1

        return round(cnt/len(truth_label),4)



    @classmethod
    #计算准确率
    def get_accuracy(self,TP,FP,FN,TN):
        res=(TP+TN)/(TP+FP+FN+TN)
        return round(res,4)


    @classmethod
    #计算召回率
    def get_recall(self,TP,FN):
        res=TP/(TP+FN)
        return round(res,4)


    @classmethod
    #计算精确率
    def get_precision(self,TP,FP):
        res=TP/(FP+TP)
        return round(res,4)
    @classmethod
    #计算F1
    #b>1时，召回率有更大影响
    #b<1时，精准率有更大影响
    def get_F1(self,TP,FP,FN,b=1):
        p=self.get_precision(TP,FP)
        r=self.get_recall(TP,FN)
        # F1=2*(p*r/(p+r))
        F1=(1+b**2)*p*r/(((b**2)*p)+r)
        return F1 

    @classmethod
    #计算方差
    def get_var(self,data):
        return np.var(data)

    @classmethod
    #计算标准差
    def get_std_deviation(self,data):
        return np.std(data)

    @classmethod
    #均方误差
    def get_mse(self,actual_values,predicted_values):
        mse = np.mean((np.array(actual_values) - np.array(predicted_values))**2)
        return mse 
    
    @classmethod
    #均方根误差
    def get_rmse(self,actual_values,predicted_values):
        mse = np.mean((np.array(actual_values) - np.array(predicted_values))**2)
        return math.sqrt(mse)
    
    @classmethod 
    #求决定系数
    def get_R2(self,actual_values,predicted_values):
        r2 = r2_score(actual_values, predicted_values)
        return r2
    
    @classmethod 
    #相对误差
    def get_relative_error(self,actual_values,predicted_values):
        return np.abs((actual_values - predicted_values) / actual_values) * 100

    @classmethod
    #相对平均误差
    def get_relative_mean_error(self,actual_values,predicted_values):
        return np.mean(self.get_relative_error(actual_values,predicted_values))
    
    @classmethod 
    #相对均方误差
    def  get_relative_mse(self,actual_values,predicted_values):
        mean_squared_error = np.mean((actual_values - predicted_values) ** 2)
        relative_mean_squared_error = mean_squared_error / np.mean(actual_values) * 100
        return relative_mean_squared_error
    
    @classmethod
    #相对均方根误差
    def get_relative_rmse(self,actual_values,predicted_values):
        rmse=self.get_rmse(actual_values,predicted_values)
        rrmse=rmse/np.mean(actual_values)
        return rrmse 
    
    @classmethod
    #绘制ROC曲线
    def draw_roc(self,actual_values,predicted_scores):
        #假正率（FPR）和真正率（TPR）
        fpr,tpr,thresholds=roc_curve(actual_values,predicted_scores)
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    #计算得到ROC曲线的AUC
    def get_auc(self,actual_values,predicted_scores):
        #假正率（FPR）和真正率（TPR）
        fpr, tpr, thresholds = roc_curve(actual_values, predicted_scores)
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        return roc_auc

    @classmethod 
    #绘制概率曲线
    def __cost_curve(self,y_true, y_pred_prob, thresholds):
        costs = []
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            cost = self.__calculate_cost(y_true, y_pred)
            costs.append(cost)
        return costs

    def __calculate_cost(self,y_true, y_pred):
        # 自定义代价函数，根据实际情况进行修改
        # 这里使用简单的代价函数：误分类样本的数量
        return np.sum(y_true != y_pred)
    def draw_cost(self,actual_values,predicted_scores):
        thresholds = np.linspace(0, 1, 100)
        costs=self.__cost_curve(actual_values,predicted_scores,thresholds)
        # 绘制代价曲线
        plt.plot(thresholds, costs)
        plt.xlabel('Threshold')
        plt.ylabel('Cost')
        plt.title('Cost Curve')
        plt.show()

        
#数学建模常用方法类
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