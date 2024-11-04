import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
import time
import os
import keras.backend as K
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.layers import Layer
from tensorflow_addons.layers import CRF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
start_time = time.time()
# GPU分配说明
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)#避免显存爆炸导致程序崩溃

# 读取训练数据
with open('./data/processed.txt', 'r', encoding='gbk') as f:
    data = f.read().split('\n\n')
'''
# 把数据集打乱
np.random.seed(42)
np.random.shuffle(data)
'''
# 划分数据集为训练集和测试集
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
test_data = data[train_size:]

# 构建词库表和标签表
word2id = {'PAD': 0, 'UNK': 1}  # 词库表
tag2id = {'PAD': 0}  # 标签表

for sentence in train_data:
    for word_tag in sentence.split('\n'):
        if len(word_tag.split()) == 2:
            word, tag = word_tag.split()
            if word not in word2id:
                word2id[word] = len(word2id)
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)

# 将数据转化为id序列
num_tags = len(tag2id)
train_X, train_Y = [], []

for sentence in train_data:
    sentence_X, sentence_Y = [], []
    for word_tag in sentence.split('\n'):
        if len(word_tag.split()) == 2:
            word, tag = word_tag.split()
            sentence_X.append(word2id.get(word, word2id['UNK']))
            sentence_Y.append(tag2id[tag])
    if sentence_X and sentence_Y:
        train_X.append(sentence_X)
        train_Y.append(sentence_Y)

# 转换为 NumPy 数组
train_X_len = [len(sentence) for sentence in train_X]
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post')

train_Y = tf.keras.preprocessing.sequence.pad_sequences(train_Y, padding='post')
train_Y = to_categorical(train_Y, num_classes=len(tag2id))

# 使用 TensorFlow Dataset API 迭代数据
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X)).batch(batch_size)

# 定义BiLSTM+CRF模型
class CRF(tf.keras.layers.Layer):
    def __init__(self, num_tags, name='crf', **kwargs):
        super(CRF, self).__init__(name=name, **kwargs)
        self.num_tags = num_tags


    def build(self, input_shape):
        self.transition_params = self.add_weight(
            name='transitions',
            shape=(self.num_tags, self.num_tags),
            initializer='glorot_uniform',
            trainable=True)
        super(CRF, self).build(input_shape)

    def get_loss(self, targets, scores):
        mask = tf.math.not_equal(targets, 0)
        loss = self.loss_function(targets, scores)
        masked_loss = tf.boolean_mask(loss, mask)
        return tf.reduce_mean(masked_loss)

    def call(self, inputs, sequence_lengths=None, targets=None, training=None, mask=None):
        if sequence_lengths is None:
            sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (tf.shape(inputs)[1])
        if mask is None:
            mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1], dtype=tf.float32)

        scores = inputs
        tags = None

        if targets is not None:
            # 将targets进行one-hot编码，并乘以mask来把填充的标签的损失去掉
            targets_one_hot = tf.one_hot(targets, self.num_tags)
            scores -= (1 - targets_one_hot) * 1e12
            scores += targets_one_hot * 1e12
            scores *= mask[:, :, None]

        tags, _ = tfa.text.crf_decode(scores, self.transition_params, sequence_lengths)

        output = tf.keras.backend.cast(tags, 'int32')
        if targets is not None:
            # 计算CRF的损失，并添加到总的损失中
            loss = self.get_loss(targets, scores)
            self.add_loss(loss)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.num_tags,)

    def get_config(self):
        config = {
            'num_tags': self.num_tags,
            'name': self.name
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_loss(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, 0)  # 非填充标签的掩码
        log_likelihood, _ = tfa.text.crf_log_likelihood(y_pred, y_true, self.sequence_lengths, self.transition_params)
        loss = -tf.reduce_mean(log_likelihood)
        masked_loss = tf.boolean_mask(loss, mask)
        return masked_loss




tf.keras.utils.register_keras_serializable('CustomCRF')(CRF)
#serialize.register_object_type('CustomCRF', CRF, crf_layer_from_config)

# 设置模型超参数
vocab_size = len(word2id)
num_tags = len(tag2id)
embedding_dim = 50
hidden_dim = 128
epochs = 10



# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,), dtype='int32'),
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True)),
    tf.keras.layers.Dense(units=num_tags),
    CRF(num_tags)
])

# 编译模型
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.layers[-1].loss_function = loss_function  # 将损失函数应用于CRF层
model.compile(optimizer=optimizer, run_eagerly=True)

# 训练模型
history = model.fit(train_dataset, epochs=epochs)


# 在测试集上评估模型
test_X, test_Y = [], []

for sentence in test_data:
    sentence_X, sentence_Y = [], []
    for word_tag in sentence.split('\n'):
        if len(word_tag.split()) == 2:
            word, tag = word_tag.split()
            sentence_X.append(word2id.get(word, word2id['UNK']))
            sentence_Y.append(tag2id[tag])
    if sentence_X and sentence_Y:
        test_X.append(sentence_X)
        test_Y.append(sentence_Y)

# 转换为 NumPy 数组
test_X_len = [len(sentence) for sentence in test_X]
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post')
test_Y = tf.keras.preprocessing.sequence.pad_sequences(test_Y, padding='post')

# 预测标签
y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis=-1)

# 使用 CRF 解码
crf_layer = model.layers[-1]
sequence_lengths = tf.constant(test_X_len, dtype=tf.int32)
pred_tags = crf_layer(tf.constant(y_pred), sequence_lengths)

# 计算准确率
accuracy = np.mean(np.equal(test_Y, pred_tags))
print('Accuracy on test set:', accuracy)

end_time = time.time()
print("Total time:", end_time - start_time, "seconds")