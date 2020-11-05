import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
plt.style.use('seaborn')
sns.set(font_scale=2)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
TRAIN_PATH = "D:/workspace/ML/nlp"
TEST_PATH = "D:/workspace/ML/nlp"
train_labled = pd.read_csv('nCoV_100k_train.labled.csv')
df_train, df_test = train_test_split(train_labled, test_size=0.1, random_state=2020)
df_train.head()

df_train.info(memory_usage='deep')
df_train['情感倾向'].value_counts()  #0:51842 1:22871 -1:15207

# 数据清洗
df_train = df_train[df_train['情感倾向'].isin(['-1','0','1'])]
df_train['情感倾向'] = df_train['情感倾向'].astype(np.int32)
df_train.info(memory_usage='deep')
df_train.head()
# 标签总体分布
df_train['情感倾向'].value_counts()/df_train['情感倾向'].count()
(df_train['情感倾向'].value_counts()/df_train['情感倾向'].count()).plot.bar()
# 时间对于情感的影响
df_train['time'] = pd.to_datetime('2020年' + df_train['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
df_train['date'] = df_train['time'].dt.date
date_influence = df_train.groupby(['date','情感倾向'],as_index=False).count()
sns.relplot(x="date", y="微博id", kind="line", hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博数量分布图',fontsize=15)
plt.show()

date_influence = date_influence.merge(df_train.groupby('date',as_index=False)['情感倾向'].count().rename(columns={'情感倾向':'weibo_count'}),how='left',on='date')
date_influence['weibo_rate'] = date_influence['微博id']/date_influence['weibo_count']
sns.relplot(x="date", y="weibo_rate", kind="line", hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博情感占比分布图',fontsize=15)
plt.show()

# 微博长度
# 基于字的长度
df_train['char_length'] = df_train['微博中文内容'].astype(str).apply(len)
np.percentile(df_train['char_length'].tolist(),75)
np.percentile(df_train['char_length'].tolist(),99)

# 基于词的长度
import pkuseg
seg = pkuseg.pkuseg() 
text = seg.cut('我爱北京天安门') # 进行分词
print(text)
df_train['word_length'] = df_train['微博中文内容'].astype(str).apply(lambda x: len(seg.cut(x)))
np.percentile(df_train['word_length'].tolist(),99)  #99

# 微博图片长度分析
df_train['pic_len'] = df_train['微博图片'].apply(lambda x: len(eval(x)))
df_train['pic_len'].value_counts()
df_train['pic_len'].value_counts().plot.bar()
plt.title('pic_len(target)')
sns.countplot(x='pic_len', hue='情感倾向',data=df_train)
plt.show()
# 微博视频长度分析
df_train['video_len'] = df_train['微博视频'].apply(lambda x: len(eval(x)))
df_train['video_len'].value_counts()
df_train['video_len'].value_counts().plot.bar()
plt.title('video_len(target)')





import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K

import os
from transformers import *

print(tf.__version__)

# 文件读取
TRAIN_PATH = './data/'
TEST_PATH = './data/'
BERT_PATH = './bert_master/'
MAX_SEQUENCE_LENGTH = 140
input_categories = '微博中文内容'
output_categories = '情感倾向'

df_train = pd.read_csv(TRAIN_PATH+'train.csv')
df_train = df_train[df_train[output_categories].isin(['-1','0','1'])]
df_test = pd.read_csv(TEST_PATH+'test.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer.encode_plus("深度之眼",
        add_special_tokens=True,
        max_length=20,
        truncation_strategy= 'longest_first')
# {'input_ids': [101, 3918, 2428, 722, 4706, 102],   #Token序列
#  'token_type_ids': [0, 0, 0, 0, 0, 0],             #Segment序列
#  'attention_mask': [1, 1, 1, 1, 1, 1]}             #Position序列 会在bert里面embedding

def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
        """默认返回input_ids,token_type_ids,attention_mask"""
        # 使用tokenizer接口  将文本进行编码 生成一个字典 字典中包含三个元素
        # instance 文本
        inputs = tokenizer.encode_plus(instance,
                                       add_special_tokens=True,
                                       max_length=max_sequence_length,
                                       truncation_strategy='longest_first')
        # 将编码后的内容取出来
        input_ids = inputs["input_ids"]
        input_masks = inputs["attention_mask"]
        input_segments = inputs["token_type_ids"]
        padding_length = max_sequence_length - len(input_ids)
        # 填充
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        return [input_ids, input_masks, input_segments]

# 将所有的文本进行保存
def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
        input_ids, input_masks, input_segments = [], [], []
        for instance in tqdm(df[columns]):
                ids, masks, segments = _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

                input_ids.append(ids)
                input_masks.append(masks)
                input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32),
                np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)
                ]
# 将训练集和测试集同时变成三个序列
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

# 标签类别转化
# 改为从0开始 变为 0 1 2 只需要加一
def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int) + 1)
outputs = compute_output_arrays(df_train, output_categories)

def create_model():
        # 三个序列作为输入
        input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
        input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
        input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
        # 导入bert模型
        # output_hidden_states   transformer中的每一层都可以取出来
        config = BertConfig.from_pretrained('./bert-master/bert-base-chinese-config.json', output_hidden_states=True)
        bert_model = TFBertModel.from_pretrained('./bert-master/bert-base-chinese-tf_model.h5', config=config)
        # bert模型会返回三个向量
        # sequence_output  最后一层transformer的向量 (bs,140,768) batchsize 文本的长度  每一个Token的向量
        # pooler_output 通过pooling之后的到的结果
        # hidden_states 12层的transformer
        sequence_output, pooler_output, hidden_states = bert_model(input_id, attention_mask=input_mask,
                                                                   token_type_ids=input_atn)
        # (bs,140,768)(bs,768)
        x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
        x = tf.keras.layers.Dropout(0.15)(x)
        x = tf.keras.layers.Dense(3, activation='softmax')(x)
        # 模型的定义
        model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # 定义loss 优化函数 和 metrics指标
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        return model

gkf = StratifiedKFold(n_splits=5).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))

valid_preds = []
test_preds = []
# 简单的进行分裂

for fold, (train_idx, valid_idx) in enumerate(gkf):
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = to_categorical(outputs[train_idx])

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = to_categorical(outputs[valid_idx])

    K.clear_session()
    model = create_model()
    # 进行模型训练的部分
    model.fit(train_inputs, train_outputs, validation_data= [valid_inputs, valid_outputs], epochs=2, batch_size=32)
    model.save_weights(f'bert-{fold}.h5')
    valid_preds.append(model.predict(valid_inputs))
    test_preds.append(model.predict(test_inputs))

df_test.head()

# 模型预测  1做平均 将概率相加取平均
sub = np.average(test_preds, axis=0)
sub = np.argmax(sub,axis=1)
# df_sub['y'] = sub-1
# #df_sub['id'] = df_sub['id'].apply(lambda x: str(x))
# df_sub.to_csv('test_sub.csv',index=False, encoding='utf-8')


# 要用测试集 微博id
df_sub = df_test[['微博id']]
df_sub.head()

# 将测试集写入
# 预测的时候 是将结果+1  实际写入的时候 要-1
df_sub['y'] = sub-1

df_sub.columns=['id','y']

df_sub.head()

df_sub.to_csv('test_sub.csv',index=False, encoding='utf-8')
