


from HaoChiUtils import DataPreprocess 
from WeiboComments import WeiboCommentCrawler as WCC


if __name__ =='__main__':

    # ================DataPreprocess 例子==================
    # DP.text_process("DataSet.tsv","Clean_data.tsv")
    # DP=DataPreprocess()
    # DP.text_process("DataSet.tsv","Clean_data.tsv")
    # res=DP.text_clean('待,处理,！的,。文本;,')
    # print(res)




    # #================WeiboCommentsCrawler例子爬取某一微博下面评论的例子=================
    # # 第一部分参数
    # # 请求头
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Mobile Safari/537.36",
    #     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    #     "accept-encoding": "gzip, deflate, br",
    # }


    # # 一些我们在网页上需要切入的参数
    # # uid = '6716429228'  # 微博的value值和用户的id是相同的
    # # l_fid = '2304136716429228'
    # # container_id = '107603{}'.format(uid)

    # # 实验性部分，进行批量处理
    # uid = ['6716429228', '5201820020']
    # l_fid = ['2304136716429228', '1076035201820020']
    # container_id = ['107603{}'.format(i) for i in uid]
    # min_since_id = ''
    # comments_nums=5
    # # 摘取语句
    # for i in range(len(uid)):
    #     WCC.fetch_file(comments_nums, headers, uid[i], l_fid[i], container_id[i], min_since_id)



    #==========================================================================================

    #====================爬取某一用户微博的例子===============================

    # headers = {
    #     "User-Agent": UserAgent().random,
    #     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    #     "accept-encoding": "gzip, deflate, br",
    # }

    min_since_id = ''
    uid = ['6716429228', '5201820020']
    l_fid = ['107603{}'.format(i) for i in uid]
    container_id = ['107603{}'.format(i) for i in uid]

    # 会使用到的存储空间
    text_data = []
    time_s = ''

    # 执行摘取语句部分
    for i in range(len(uid)):
        text_data, time_s = WCC.get_data( min_since_id, uid[i], l_fid[i], container_id[i])
        # print(text_data)
        WCC.save_file(WCC.get_user_name( uid[i], container_id[i]) + '_' + time_s, text_data[::-1])
        # WCC.save_file(text_data[::-1])
