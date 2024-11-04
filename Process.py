# process.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, input_file, output_file, processed_file):
        self.input_file = input_file
        self.output_file = output_file
        self.processed_file = processed_file
        self.df = pd.read_csv(self.input_file, encoding='gbk')
        self.tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

    def fill_missing_values(self):
        self.df = self.df.fillna('')

    def preprocess_data(self):
        data = []
        for i, row in self.df.iterrows():
            sentence = []
            tags = []
            for col in self.df.columns:
                if col == '招标人':
                    entity_tag = 'B-ORG'
                elif col == '中标人':
                    entity_tag = 'B-PER'
                elif col == '中标金额':
                    entity_tag = 'B-LOC'
                elif col == '中标时间':
                    entity_tag = 'B-LOC'
                else:
                    entity_tag = 'O'

                tokens = row[col].split()
                for j, token in enumerate(tokens):
                    sentence.append(token)
                    if j == 0:
                        tags.append(entity_tag)
                    else:
                        tags.append('I' + entity_tag[1:])
            data.append((sentence, tags))
        return data

    def save_processed_data(self, data):
        with open(self.output_file, 'w', encoding='gbk') as f:
            for sentence, tags in data:
                for i in range(len(sentence)):
                    f.write(sentence[i] + ' ' + tags[i] + '\n')
                f.write('\n')
        print(f"Data processed and saved to {self.output_file}")
    def convert_to_numpy(self, data):
        X = np.array([sentence for sentence, _ in data], dtype=object)
        y = np.array([[self.tag2id[tag] for tag in tags] for _, tags in data], dtype=object)
        return X, y

    def save_npz(self, X_train, y_train, X_test, y_test):
        np.savez(self.processed_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 tag2id=self.tag2id, id2tag=self.id2tag)

    def process(self):
        self.fill_missing_values()
        data = self.preprocess_data()
        self.save_processed_data(data)

        X, y = self.convert_to_numpy(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.save_npz(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    processor = DataProcessor(
        input_file='./data/merge.csv',
        output_file='./data/processed.txt',
        processed_file='./data/processed.npz'
    )
    processor.process()
