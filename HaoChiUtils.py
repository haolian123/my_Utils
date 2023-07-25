# 导入所需的库
import re
import csv
import jieba
import emoji
from opencc import OpenCC
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#定义数据预处理类 by CXC
#包含 清洗文本、将清洗后的文本存入文件方法
class DataPreprocess:

    # 指定的停用词
    __stop_terms = ["展开", "全文", "转发", "显示原图", "原图","显示地图"]

    #停用词表
    __stopwords = []

    def __init__(self,stopwords_file_path = "hit_stopwords.txt") :
        # 加载停用词列表
        
        with open(stopwords_file_path, "r", encoding="utf-8") as stopwords_file:
            for line in stopwords_file:
                self.__stopwords.append(line.strip())


    # 定义清洗文本的函数
    @classmethod
    def text_clean(self,text,has_user_id=False):
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
        zh_puncts1 = "，；、。！？（）《》【】"
        URL_REGEX = re.compile(
            r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' +
            zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
            r'[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',re.IGNORECASE)
        
        # 使用正则表达式去除URL
        text = re.sub(URL_REGEX, "", text)
        
        # 使用正则表达式去除@用户和回复标记
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)", " ", text)
        
        # 使用正则表达式去除表情符号
        text = re.sub(r"\[\S+?\]", "", text)
        
        # 使用正则表达式去除话题标签
        text = re.sub(r"#\S+#", "", text)
        
        # 使用正则表达式去除多余的空格
        text = re.sub(r"(\s)+", r"\1", text)
        
        
        for x in self.__stop_terms:
            text = text.replace(x, "")
        
        # 去除首尾空格
        text = text.strip()
        
        # 将表情符号转换为文本描述
        text = emoji.demojize(text)
        
        # 使用结巴分词进行分词
        seg_list = list(jieba.cut(text,cut_all=False))
        
        # 去除停用词
        seg_list = [word for word in seg_list if word not in self.__stopwords]
        
        # 将分词结果拼接为字符串
        cleaned_text = ' '.join(seg_list)
        
        return cleaned_text

    #item_len 为每一行有多少个字段，默认为3，预处理字段需要在第item_len列
    @classmethod
    def text_process(self,input_file_path="DataSet.tsv", output_file_path="Clean_data.tsv",item_len=3):

        count=1
        # 打开输入文件并读取内容
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
            cleaned_lines = []
            
            # 遍历每一行数据
            for line in lines:
                line = line.strip().split('\t')
                if count%500==0:
                    print(f"已处理{count}条文本记录")
                count+=1
                # 检查列表长度是否足够
                if len(line) == item_len:
                    # 调用clean_text函数清洗第一列的文本数据，并保留其他几列数据
                    cleaned_line = [self.__clean_text(line[0])]
                    for index in range(1,len(line)):
                        cleaned_line.append(line[index])
                    cleaned_lines.append(cleaned_line)
            
            # 删去第一列内容为空的行
            cleaned_lines = [line for line in cleaned_lines if line[0]]
            
            # 打开输出文件并写入清洗后的数据，写入csv
            with open(output_file_path, "w", encoding="utf-8", newline='') as output_file:
                writer = csv.writer(output_file,delimiter='\t')
                for line in cleaned_lines:
                    writer.writerow(line)
            print(f"共有{len(cleaned_lines)}条记录！")
            # # 输出提示信息
            # print("修改后的内容已写入新文件。")



#数据分析类 by Hao
#包含 划分数据集、绘制训练曲线、计算标签占比方法
class DataAnalyzer:

    def __init__(self) -> None:
        pass


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

    #绘制训练过程的曲线
    @classmethod
    def draw_process(self,title='trainning acc',color='r',iters=[],data=[],label='trainning acc',png_path='plot'):
        plt.title(title, fontsize=24)
        plt.xlabel("iter", fontsize=20)
        plt.ylabel(label, fontsize=20)
        plt.plot(iters, data,color=color,label=label) 
        plt.legend()
        plt.grid()
        plt.savefig(png_path+'/'+label+'.png')
        # plt.show()


    #计算标签占比，输入为预测结果的列表,输出为 标签:占比
    @classmethod
    def calculate_label_proportions(predictions):
        predictions_dict={}
        total_cnt=0
        for prediction in predictions:
            if prediction not in  predictions_dict.keys():
                 predictions_dict[prediction]=0
            predictions_dict[prediction]+=1
            total_cnt+=1
        res_dict={}

        for key,value in  predictions_dict.items():
            res_dict[key]={}
            res_dict[key]['total']=value
            proportion=round(value/total_cnt*100,1)
            print(proportion)
            res_dict[key]['proportion']=rf'{proportion}%'
        return res_dict