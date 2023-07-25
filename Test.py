


from HaoChiUtils import DataPreprocess as DP



if __name__ =='__main__':

#    DP.text_process("DataSet.tsv","Clean_data.tsv",item_len=3)
    res=DP.text_clean("待处理的文本")
    print(res)