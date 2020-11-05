# 基于百度aistudio,GPU环境
import pandas as pd
train = pd.read_csv('data/data24278/train.csv', engine ='python')
unlablel = pd.read_csv('data/data24278/train_label.csv', engine ='python')
train.head()

print(train.shape)
print(train.columns)
print(unlablel.shape)
print(unlablel.columns)
# 标签分布
%matplotlib inline
train['情感倾向'].value_counts(normalize=True).plot(kind='bar');

# 微博长度
train['微博中文内容'].str.len().describe()

# 划分验证集，保存格式  text[\t]label
from sklearn.model_selection import train_test_split

train_labled = train[['微博中文内容', '情感倾向']]
train, valid = train_test_split(train_labled, test_size=0.2, random_state=2020)
train.to_csv('/home/aistudio/data/data24278/train.txt', index=False, header=False, sep='\t')
valid.to_csv('/home/aistudio/data/data24278/valid.txt', index=False, header=False, sep='\t')

# 自定义数据集
import os
import codecs
import csv

from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

class MyDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "/home/aistudio/data/data24278"
        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.txt",
            dev_file="valid.txt",
            test_file="valid.txt",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # 数据集类别集合
            label_list=["-1", "0", "1"])
			
dataset = MyDataset()
for e in dataset.get_train_examples()[:3]:
    print("{}\t{}\t{}".format(e.guid, e.text_a, e.label))
	

# 加载模型
import paddlehub as hub
module = hub.Module(name='bert_chinese_L-12_H-768_A-12')

# 构建Reader
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path(),
    max_seq_len=140)

# finetune策略
strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5)
	
# finetune策略
strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5)
	
# 运行配置
config = hub.RunConfig(
    use_cuda=True,
    num_epoch=1,
    checkpoint_dir="model",
    batch_size=32,
    eval_interval=100,
    strategy=strategy)

# Finetune Task
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=140)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=["f1"])
		
# finetune
run_states = cls_task.finetune_and_eval()

# test
import numpy as np
inv_label_map = {val: key for key, val in reader.label_map.items()}
# Data to be prdicted
data = valid[['微博中文内容']].fillna(' ').values.tolist()
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
# 生成预测结果
proba = np.vstack([r[0] for r in results])
prediction = list(np.argmax(proba, axis=1))
prediction = [inv_label_map[p] for p in prediction]
print(len(prediction))
valid['y'] = prediction
print(valid.shape)
from sklearn.metrics import confusion_matrix
confusion_matrix(valid['情感倾向'].astype(str), valid['y'].astype(str))
from sklearn.metrics import f1_score, precision_score, recall_score
f1 = f1_score(valid['情感倾向'].astype(str), valid['y'].astype(str), average='weighted' )
p = precision_score(valid['情感倾向'].astype(str), valid['y'].astype(str), average='weighted')
r = recall_score(valid['情感倾向'].astype(str), valid['y'].astype(str), average='weighted')
print(p,r,f1)

# 生成预测结果
data = unlablel[['微博中文内容']].fillna(' ').values.tolist()
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
proba = np.vstack([r[0] for r in results])
prediction = list(np.argmax(proba, axis=1))
prediction = [inv_label_map[p] for p in prediction]
unlablel['y'] = prediction
unlablel.head()
unlablel.to_csv('/home/aistudio/data/data24278/label.csv')
