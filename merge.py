#merge.py

import os
import pandas as pd
import chardet

class CSVMerger:
    def __init__(self, folder_path, output_file, usecols):
        self.folder_path = folder_path
        self.output_file = output_file
        self.usecols = usecols
        self.all_data = []

    def merge_csv_files(self):
        # 获取文件夹中所有csv文件的文件名
        file_list = os.listdir(self.folder_path)
        csv_files = [file_name for file_name in file_list if file_name.endswith('.csv')]

        for file_name in csv_files:
            # 构造每个csv文件的路径
            file_path = os.path.join(self.folder_path, file_name)
            try:
                # 探测文件编码格式
                with open(file_path, 'rb') as f:
                    encoding = chardet.detect(f.read())['encoding']
                # 读取csv文件的数据
                temp_df = pd.read_csv(file_path, encoding=encoding, usecols=self.usecols)
                self.all_data.append(temp_df)
                print(f"Reading {file_path}...")
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                print(f"Error reading {file_path}. Skipping...")

        # 将所有数据进行纵向合并
        self.combine_data()

    def combine_data(self):
        # 将所有数据合并
        result_df = pd.concat(self.all_data)

        result_df.replace(['无', '未提供', '未提及'], '', inplace=True)
        # NaN替换为空值
        result_df.fillna('', inplace=True)
        # 删除空行
        result_df.dropna(thresh=1, inplace=True)

        # 将结果保存到csv文件中
        result_df.to_csv(self.output_file, encoding='gbk', index=False)
        print(f"Data merged and saved to {self.output_file}")

if __name__ == '__main__':
    # 示例用法
    merger = CSVMerger(folder_path='./data/nlp-csv/', output_file='./data/merge.csv', usecols=[1, 2, 3, 4])
    merger.merge_csv_files()