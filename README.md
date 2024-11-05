# 这里是一个基于BiLSTM+CRF层实现的实体抽取实战

## 安装依赖：

```shell
cd path\to\your\project
pip install -r requirements.txt
```

## 如何运行：

```shell
python main.py
```



## 实验思路：

本次的数据包含30个csv文件，字段包含行数、招标人、中标人、中标金额、中标时间五列。

想要实现实体提取，首先需要将数据进行预处理，将处理好的数据放入BILSTM+CRF模型中训练，并且在训练后对正确率进行测试。

以下是文章脉络：

- 数据预处理
- 模型选择
- 模型训练
- 结果展示

## 数据预处理：

### 合并分散的CSV：

首先我们初始化实例，这个构造函数过接受 `folder_path`、`output_file` 和 `usecols` 三个参数，初始化了 `CSVMerger` 类实例的一些核心配置（如文件夹路径、输出文件路径、要读取的列）。同时，它还创建了一个空列表 `all_data`，用于存储后续读取的 CSV 文件数据。

```python
def __init__(self, folder_path, output_file, usecols):
    self.folder_path = folder_path
    self.output_file = output_file
    self.usecols = usecols
    self.all_data = []
```

由于搜集到的数据是多个csv并且行和列有固定格式，所以我们直接调用 pandas 库对csv进行合并。提取列名为招标人、中标人、中标金额、中标时间这四列下的所有数据。

合并时考虑到不同csv可能使用不同的编码格式，我们使用 chardet 库对每个csv进行编码格式的检测，并且以原本的格式读取。

对合并的数据进行简单的预处理。将NaN替换为空值并且删除空行。最后保存为编码格式为"gbk"的csv。

```python
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
                print(f"Failed reading {file_path}. Skipping...")

        # 将所有数据进行纵向合并
        self.combine_data()
```

```python
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
```

### 转化标签：

在上一步我们对数据进行了合并并且进行了简单的预处理。现在我们需要对csv中的四个实体进行NER标注。为了实现在BILSTM模型中训练，我们需要将已经打好的标签转化为数字标签。并且将样本和标签对应。

同样的，先初始化实例。该构造函数初始化了类实例所需的路径（输入文件、输出文件和处理后的文件）。

使用 `pandas` 读取输入的 CSV 文件并存储到 `self.df` 中，方便后续的处理。

创建了两个字典：`tag2id` 用于标签到 ID 的映射，`id2tag` 用于 ID 到标签的反向映射，常用于处理标签化数据，尤其是命名实体识别（NER）任务中。

```python
def __init__(self, input_file, output_file, processed_file):
    self.input_file = input_file
    self.output_file = output_file
    self.processed_file = processed_file
    self.df = pd.read_csv(self.input_file, encoding='gbk')
    self.tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    self.id2tag = {v: k for k, v in self.tag2id.items()}
```

这段代码定义了两个Python字典，用于将命名实体标签转换为数字标签以进行机器学习模型的训练和预测。

tag2id字典将命名实体标签映射到相应的数字标签，其中'O'代表非命名实体标签，'B-PER'和'I-PER'代表人名实体的开始和内部标记，'B-ORG'和'I-ORG'代表组织机构名实体的开始和内部标记，'B-LOC'和'I-LOC'代表位置名实体的开始和内部标记。数字标签从0开始，非命名实体标签为0，其余标签从1开始递增。

而id2tag字典则是将数字标签转换回相应的命名实体标签，以便在训练后对预测结果进行解码。

```python
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
```


这段代码是将一个Pandas数据框（df）中的每个数据样本（即每一行）转换为适合输入模型的格式。具体而言，它将每个数据样本表示为一个包含两个列表的元组，其中第一个列表是数据样本的分词序列，第二个列表是与分词序列对应的命名实体标签序列。

在该函数中，首先遍历每个数据样本，然后遍历该样本的每一列。对于不同的列，函数将其值的分词序列转换为相应的命名实体标签序列。例如，如果列名为“招标人”，则该列的实体标签被设置为“B-ORG”（即开始的组织机构实体），并且该列的每个分词都被分配为该实体的内部标记（即“I-ORG”）。

函数将转换后的数据样本存储在一个列表（data）中并返回。调用函数的代码将返回的列表（data）赋值给变量data并输出一份编码为gbk，列之间用空格分隔的txt文件。

```python
    def save_processed_data(self, data):
        with open(self.output_file, 'w', encoding='gbk') as f:
            for sentence, tags in data:
                for i in range(len(sentence)):
                    f.write(sentence[i] + ' ' + tags[i] + '\n')
                f.write('\n')
        print(f"Data processed and saved to {self.output_file}")
```

process方法将DataProcessor类的各个方法串联并且调用，完成了数据处理。

```python
    def process(self):
        self.fill_missing_values()
        data = self.preprocess_data()
        self.save_processed_data(data)

        X, y = self.convert_to_numpy(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.save_npz(X_train, y_train, X_test, y_test)
```



## 模型选择：

训练模型我们选择BILSTM(双向长短期记忆网络)+CRF(条件随机场)相结合的模型，BILSTM-CRF。

BILSTM-CRF是一种用于命名实体识别的序列标注模型。有以下特点：

1. 上下文信息：BILSTM能够捕捉序列数据中的上下文信息，包括前面和后面的词语，从而更好地理解语境，提高预测的准确性。
2. 不同长度的序列：BILSTM可以应对不同长度的序列，因为它是基于序列的输入进行处理的，因此对于不同长度的序列也能够产生较为稳定的结果。
3. CRF模型：CRF模型可以将BILSTM输出的概率分布转换为标签序列，考虑到标签之间的相关性，从而更好地处理标注结果的一致性和准确性。
4. 鲁棒性：BILSTM-CRF模型通常在命名实体识别任务中表现优异，并且在处理噪声和错误的情况下表现鲁棒性强，这在现实场景中具有实际应用价值。

综上，BILSTM-CRF模型能够从序列数据中提取上下文信息并产生准确且一致的标注结果，因此在序列标注任务中具有广泛的应用前景。

## 模型训练：

### 调包以及环境配置

想要实现GPU计算首先需要导入所需的包，并且检查GPU是否可用

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import os
# GPU分配说明
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)#避免显存爆炸导致程序崩溃
```

其中`tf.config.experimental.set_memory_growth(physical_devices[0], True)`这段函数是用于设置 GPU 内存动态增长的，避免 GPU 内存被全部占满导致程序崩溃。该函数的参数是一个物理设备对象以及一个布尔值，表示是否启用内存动态增长。

### 数据划分和传入

首先将数据载入，并且划分90%训练集，10%测试集。

```python
# 读取训练数据
with open('./data/processed.txt', 'r', encoding='gbk') as f:
    data = f.read().split('\n')
# 划分数据集为训练集和测试集
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
test_data = data[train_size:]
```

由于预处理的数据为两列，左边是实体，右边是词性，所以建立词汇表和标签表。词汇表是一个字典，将每个词映射为一个整数ID；标签表也是一个字典，将每个标签映射为一个整数ID。

```python
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
```

对于每个句子，将每个单词和标签分别转化为词库表和标签表中的对应编号。如果当前单词不在词库表中，则使用`'UNK'`对应的编号。这样得到的结果是两个列表，分别记录每个句子的单词和标签编号。

```python
# 转换为 NumPy 数组
train_X_len = [len(sentence) for sentence in train_X]
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post')

train_Y = tf.keras.preprocessing.sequence.pad_sequences(train_Y, padding='post')
train_Y = to_categorical(train_Y, num_classes=len(tag2id))
```

在得到每个句子的单词和标签编号列表后，使用`pad_sequences`函数将每个列表转化为指定长度的NumPy数组，其中长度为每个句子的最大长度，短句子会在后面补0，即`'PAD'`的编号。然后使用`to_categorical`函数对标签进行one-hot编码。

```python
# 使用 TensorFlow Dataset API 迭代数据
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X)).batch(batch_size)
```

使用`from_tensor_slices`函数将训练数据拆分成小批量数据，并设置一个batch_size。最后使用`shuffle`函数进行数据打乱，实现更好的训练效果。

### 模型搭建

```python
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
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(
                scores, targets, sequence_lengths, self.transition_params)
            self.add_loss(-tf.reduce_mean(log_likelihood))

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
```

这段代码定义了一个CRF层，用于将BiLSTM的输出转化为序列标注的结果。具体来说，该层的主要功能是进行序列标注的解码，同时计算序列标注的损失。

在实现上，该层继承了tf.keras.layers.Layer类，并重写了call()方法。在call()方法中，通过调用tensorflow-addons库中的tfa.text.crf_decode()方法实现了序列标注的解码，得到标注结果。同时，如果提供了targets，该层会调用tfa.text.crf_log_likelihood()方法计算CRF的损失，并将其添加到总的损失中。最终返回标注结果。

在类的定义中，还实现了compute_output_shape()和get_config()方法，用于返回该层的输出形状和配置信息。

```python
# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,), dtype='int32'),
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True)),
    tf.keras.layers.Dense(units=num_tags),
    CRF(num_tags)
])

```


这段代码定义了一个基于双向LSTM和CRF的序列标注模型。具体来说，它包括以下几个部分：

1. `Input`层：接收输入数据，该层指定了输入的形状和数据类型。
2. `Embedding`层：将输入的整数序列映射为密集向量表示，这里的输入是字或词在词汇表中的索引，每个索引被映射为一个embedding_dim维的向量。
3. 双向LSTM层：LSTM（长短时记忆网络）是一种循环神经网络，双向LSTM在LSTM的基础上加入了从后向前的时间流，可以更好地捕捉序列中的上下文信息。
4. 全连接层：将LSTM层的输出通过一个全连接层转换为num_tags维的向量，其中num_tags是标签的数量。
5. CRF层：CRF（条件随机场）是一种常用于序列标注的方法，它将标签之间的依赖关系建模为一个图，并通过动态规划算法求解给定输入序列的最优标签序列。这个实现基于TensorFlow Addons的`CRF`层，它计算输入序列上的CRF损失，并输出最优标签序列作为模型的输出。

最终，这个模型接收一个整数序列作为输入，并输出对应的标签序列。模型的训练目标是最小化CRF损失。

```python
# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=model.layers[-1].loss)
# 训练模型
history = model.fit(train_dataset, epochs=epochs)
```

这段代码编译了模型并使用Adam优化器，设置了学习率为0.001。模型的损失函数使用了CRF层的损失函数。接着，使用`fit`函数训练模型，使用训练集数据集`train_dataset`，训练轮数为`epochs`。训练过程中会返回历史损失值的记录，存储在`history`变量中。

### 模型检测：

```python
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
```

首先，将测试集中的句子逐一处理，对每个句子中的单词和标签进行分割，并将单词和标签分别映射为整数，形成句子的输入和标签的输出。这里使用了一个词典 word2id 来将单词映射为整数，如果单词不在词典中，则将其映射为一个特殊的未知词（UNK）。同样地，使用一个标签词典 tag2id 将标签映射为整数。

```python
# 转换为 NumPy 数组
test_X_len = [len(sentence) for sentence in test_X]
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post')

test_Y = tf.keras.preprocessing.sequence.pad_sequences(test_Y, padding='post')
test_Y = to_categorical(test_Y, num_classes=len(tag2id))
```

接下来，将处理后的测试数据转换为 NumPy 数组，通过调用 Keras 的 pad_sequences() 函数，将输入和输出的长度都进行填充，以满足模型的输入要求。为了让输出的标签符合模型的要求，使用 to_categorical() 函数将其转换为独热编码形式。

```python
# 预测标签
y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis=-1)

# 使用 CRF 解码
crf_layer = model.layers[-1]
sequence_lengths = tf.constant(test_X_len, dtype=tf.int32)
pred_tags = crf_layer(tf.constant(y_pred), sequence_lengths)
```

这段代码使用定义的CRF层解码预测结果，得到最终的标签。

具体来说，`model.predict(test_X)` 用于对测试集进行预测，返回的是模型对测试集每个位置的标签的预测概率。其中 `test_X` 是测试集输入数据的序列，`test_X_len` 是测试集输入数据序列的长度。通过 `np.argmax(y_pred, axis=-1)` 对每个位置的预测概率取最大值，得到预测标签序列 `y_pred`。

然后，使用定义的 CRF 层进行解码，即 `crf_layer(tf.constant(y_pred), sequence_lengths)`。其中 `y_pred` 是上一步得到的预测标签序列，`sequence_lengths` 是测试集输入数据序列的长度。这一步返回的是 CRF 层解码后的标签序列，即最终的预测结果 `pred_tags`。

```python
# 计算准确率
accuracy = np.mean(np.equal(test_Y, pred_tags))
print('Accuracy on test set:', accuracy)
print("Total time:", end_time - start_time, "seconds")
```

最后输出测试集准确率以及程序总耗时。

## 结果展示：

```css
Epoch 1/15
48/48 [==============================] - 7s 17ms/step - loss: 0.4298 - accuracy: 0.6026
Epoch 2/15
48/48 [==============================] - 1s 17ms/step - loss: 0.2182 - accuracy: 0.7013
Epoch 3/15
48/48 [==============================] - 1s 17ms/step - loss: 0.1517 - accuracy: 0.7791
Epoch 4/15
48/48 [==============================] - 1s 17ms/step - loss: 0.1130 - accuracy: 0.8435
Epoch 5/15
48/48 [==============================] - 1s 14ms/step - loss: 0.0773 - accuracy: 0.9575
Epoch 6/15
48/48 [==============================] - 1s 14ms/step - loss: 0.0421 - accuracy: 0.9704
Epoch 7/15
48/48 [==============================] - 1s 16ms/step - loss: 0.0242 - accuracy: 0.9728
Epoch 8/15
48/48 [==============================] - 1s 16ms/step - loss: 0.0170 - accuracy: 0.9752
Epoch 9/15
48/48 [==============================] - 1s 16ms/step - loss: 0.0132 - accuracy: 0.9804
Epoch 10/15
48/48 [==============================] - 1s 17ms/step - loss: 0.0105 - accuracy: 0.9881
Epoch 11/15
48/48 [==============================] - 1s 17ms/step - loss: 0.0082 - accuracy: 0.9928
Epoch 12/15
48/48 [==============================] - 1s 19ms/step - loss: 0.0068 - accuracy: 0.9948
Epoch 13/15
48/48 [==============================] - 1s 15ms/step - loss: 0.0056 - accuracy: 0.9962
Epoch 14/15
48/48 [==============================] - 1s 14ms/step - loss: 0.0078 - accuracy: 0.9895
Epoch 15/15
48/48 [==============================] - 1s 15ms/step - loss: 0.0036 - accuracy: 0.9971
Accuracy on test set: 0.8867924809455872
Total time: 21.525487422943115 seconds
```

<center>测试集准确率88.67%<center>


<center>程序总耗时：21.52 seconds<center>




## **TODO:**

## 模型优化：

模型容易出现过拟合现象。
