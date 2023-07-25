# HaoChiUtils.py

NLP一些工具库——by 浩池懒做

## class DataPreprocess

数据预处理类

1. text_process() :将文本清洗后存入指定路径文件中
2. text_clean()： 将一个字符串文本清洗后，返回清洗后的结果

## class DataAnalyzer

数据分析类

1. split_dataSet(): 划分数据集为测试集、验证集、训练集
2. draw_process(): 绘制训练过程的曲线
3. calculate_label_proportions(): 计算标签占比，输入为预测结果的列表,输出为 标签:占比

# hit_stopwords.txt

停用词表



# WeiboComments.py

爬取微博特定用户的评论——by汪之鱼

## class WeiboCommentCrawler

爬取微博评论的类

1. **fetch_file**(self,count=5, header=__default_headers, user_id=[], l_id=[], contain_id=[], since_id=''):

   1. headers：HTTP请求头，包含了请求的一些元数据，如用户代理、授权信息等。在发送请求时，需要将适当的请求头信息包含在其中，以便与服务器进行通信。
   2. uid[i]：微博用户的唯一标识符。每个微博用户都有一个独特的用户ID，用于标识用户的身份。
   3. l_fid[i]：微博的唯一标识符。每条微博都有一个独特的微博ID，用于标识微博的内容。
   4. container_id[i]：微博容器的唯一标识符。微博容器是一个包含微博及其相关内容的容器，如用户主页、话题页面等。通过容器ID，可以定位到特定的微博容器，从而获取相关的评论信息。
   5. min_since_id：最小的评论ID。通过设置最小的评论ID，可以筛选出大于该ID的评论，以获取最新的评论内容。
   6. count：爬取评论的数目。
