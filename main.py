# main.py

from merge import CSVMerger
from Process import DataProcessor
from BILSTM_zhaobiao import BiLSTMModel

def main():
    # 调用merge
    merger = CSVMerger(
        folder_path='./data/nlp-csv/',
        output_file='./data/merge.csv',
        usecols=[1, 2, 3, 4]
    )
    merger.merge_csv_files()

    # 调用Process
    processor = DataProcessor(
        input_file='./data/merge.csv',
        output_file='./data/processed.txt',
        processed_file='./data/processed.npz'
    )
    processor.process()

    # 调用BiLSTM模型训练和评估
    model = BiLSTMModel(processed_file='./data/processed.txt')
    model.load_data()
    model.build_vocab()
    model.prepare_data()
    model.create_dataset()
    model.build_model()
    model.train()
    model.evaluate()

if __name__ == '__main__':
    main()
