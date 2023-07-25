


from HaoChiUtils import DataPreprocess



if __name__ =='__main__':
    dataPreprocess=DataPreprocess()
    dataPreprocess.text_process("DataSet.tsv","Clean_data.tsv",item_len=3)