# BILSTM_zhaobiao.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
start_time = time.time()
# GPU分配说明
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)#避免显存爆炸导致程序崩溃

class BiLSTMModel:
    def __init__(self, processed_file, embedding_dim=50, hidden_dim=128, epochs=15, batch_size=16):
        self.processed_file = processed_file
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.word2id = {'PAD': 0, 'UNK': 1}
        self.tag2id = {'PAD': 0}

    def load_data(self):
        with open(self.processed_file, 'r', encoding='gbk') as f:
            data = f.read().split('\n\n')
        train_size = int(len(data) * 0.9)
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]

    def build_vocab(self):
        for sentence in self.train_data:
            for word_tag in sentence.split('\n'):
                if len(word_tag.split()) == 2:
                    word, tag = word_tag.split()
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id)
                    if tag not in self.tag2id:
                        self.tag2id[tag] = len(self.tag2id)

    def prepare_data(self):
        train_X, train_Y = [], []
        for sentence in self.train_data:
            sentence_X, sentence_Y = [], []
            for word_tag in sentence.split('\n'):
                if len(word_tag.split()) == 2:
                    word, tag = word_tag.split()
                    sentence_X.append(self.word2id.get(word, self.word2id['UNK']))
                    sentence_Y.append(self.tag2id[tag])
            if sentence_X and sentence_Y:
                train_X.append(sentence_X)
                train_Y.append(sentence_Y)

        self.train_X_len = [len(sentence) for sentence in train_X]
        self.train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post')
        self.train_Y = tf.keras.preprocessing.sequence.pad_sequences(train_Y, padding='post')
        self.train_Y = to_categorical(self.train_Y, num_classes=len(self.tag2id))

    def create_dataset(self):
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_X, self.train_Y)).shuffle(len(self.train_X)).batch(self.batch_size)

    def build_model(self):
        vocab_size = len(self.word2id)
        num_tags = len(self.tag2id)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, self.embedding_dim, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True)),
            tf.keras.layers.Dense(num_tags, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        start_time = time.time()
        self.history = self.model.fit(self.train_dataset, epochs=self.epochs)
        end_time = time.time()
        print("Training time:", end_time - start_time, "seconds")

    def evaluate(self):
        test_X, test_Y = [], []
        for sentence in self.test_data:
            sentence_X, sentence_Y = [], []
            for word_tag in sentence.split('\n'):
                if len(word_tag.split()) == 2:
                    word, tag = word_tag.split()
                    sentence_X.append(self.word2id.get(word, self.word2id['UNK']))
                    sentence_Y.append(self.tag2id[tag])
            if sentence_X and sentence_Y:
                test_X.append(sentence_X)
                test_Y.append(sentence_Y)

        test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post')
        test_Y = tf.keras.preprocessing.sequence.pad_sequences(test_Y, padding='post')
        test_Y = to_categorical(test_Y, num_classes=len(self.tag2id))

        test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(self.batch_size)
        _, accuracy = self.model.evaluate(test_dataset, verbose=0)
        print('Accuracy on test set:', accuracy)

if __name__ == '__main__':
    model = BiLSTMModel(processed_file='./data/processed.txt')
    model.load_data()
    model.build_vocab()
    model.prepare_data()
    model.create_dataset()
    model.build_model()
    model.train()
    model.evaluate()
