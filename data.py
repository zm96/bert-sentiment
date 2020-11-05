## 情感分析结果进行分析，统计每日各类情感微博数量与占比
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
plt.style.use('seaborn')
sns.set(font_scale=2)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train = pd.read_csv('train.csv')
labeled = pd.read_csv('label.csv')

# labeled.drop(['Unnamed: 0'],axis=1,inplace=True)
train.head()
labeled.head()
data = pd.concat([train,labeled],axis=0)
data.shape

# 标签总体分布
data['情感倾向'].value_counts()/data['情感倾向'].count()
(data['情感倾向'].value_counts()/data['情感倾向'].count()).plot.bar()
plt.show()
# 时间对于情感的影响
data['time'] = pd.to_datetime('2020年' + data['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
data['date'] = data['time'].dt.date
date_influence = data.groupby(['date','情感倾向'],as_index=False).count()
sns.relplot(x="date", y="微博id", kind="line", hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博数量分布图',fontsize=15)
plt.show()
# 时间对于情感的影响比例
date_influence = date_influence.merge(data.groupby('date',as_index=False)['情感倾向'].count().rename(columns={'情感倾向':'weibo_count'}),how='left',on='date')
date_influence['weibo_rate'] = date_influence['微博id']/date_influence['weibo_count']
sns.relplot(x="date", y="weibo_rate", kind="line", hue='情感倾向',palette=["b", "r",'g'],data=date_influence)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('日期',fontsize=15)
plt.ylabel('数量',fontsize=15)
plt.title('微博情感占比分布图',fontsize=15)
plt.show()

# 进行数据筛选，为LDA做准备
data010102_0 = data[['微博中文内容']][((data['date']=='2020-01-01')|(data['date']=='2020-01-02'))&(data['情感倾向']==-1)]
data010102_1 = data[['微博中文内容']][((data['date']=='2020-01-01')|(data['date']=='2020-01-02'))&(data['情感倾向']==0)]
data010102_2 = data[['微博中文内容']][((data['date']=='2020-01-01')|(data['date']=='2020-01-02'))&(data['情感倾向']==1)]
data010102_0.to_csv('data010102_0.txt', index=False, header=False, sep='\t')
data010102_1.to_csv('data010102_1.txt', index=False, header=False, sep='\t')
data010102_2.to_csv('data010102_2.txt', index=False, header=False, sep='\t')

data011920_0 = data[['微博中文内容']][((data['date']=='2020-01-19')|(data['date']=='2020-01-20'))&(data['情感倾向']==-1)]
data011920_1 = data[['微博中文内容']][((data['date']=='2020-01-19')|(data['date']=='2020-01-20'))&(data['情感倾向']==0)]
data011920_2 = data[['微博中文内容']][((data['date']=='2020-01-19')|(data['date']=='2020-01-20'))&(data['情感倾向']==1)]
data011920_0.to_csv('data011920_0.txt', index=False, header=False, sep='\t')
data011920_1.to_csv('data011920_1.txt', index=False, header=False, sep='\t')
data011920_2.to_csv('data011920_2.txt', index=False, header=False, sep='\t')

data012324_0 = data[['微博中文内容']][((data['date']=='2020-01-23')|(data['date']=='2020-01-24'))&(data['情感倾向']==-1)]
data012324_1 = data[['微博中文内容']][((data['date']=='2020-01-23')|(data['date']=='2020-01-24'))&(data['情感倾向']==0)]
data012324_2 = data[['微博中文内容']][((data['date']=='2020-01-23')|(data['date']=='2020-01-24'))&(data['情感倾向']==1)]
data012324_0.to_csv('data012324_0.txt', index=False, header=False, sep='\t')
data012324_1.to_csv('data012324_1.txt', index=False, header=False, sep='\t')
data012324_2.to_csv('data012324_2.txt', index=False, header=False, sep='\t')

data020506_0 = data[['微博中文内容']][((data['date']=='2020-02-05')|(data['date']=='2020-02-06'))&(data['情感倾向']==-1)]
data020506_1 = data[['微博中文内容']][((data['date']=='2020-02-05')|(data['date']=='2020-02-06'))&(data['情感倾向']==0)]
data020506_2 = data[['微博中文内容']][((data['date']=='2020-02-05')|(data['date']=='2020-02-06'))&(data['情感倾向']==1)]
data020506_0.to_csv('data020506_0.txt', index=False, header=False, sep='\t')
data020506_1.to_csv('data020506_1.txt', index=False, header=False, sep='\t')
data020506_2.to_csv('data020506_2.txt', index=False, header=False, sep='\t')

data021718_0 = data[['微博中文内容']][((data['date']=='2020-02-17')|(data['date']=='2020-02-18'))&(data['情感倾向']==-1)]
data021718_1 = data[['微博中文内容']][((data['date']=='2020-02-17')|(data['date']=='2020-02-18'))&(data['情感倾向']==0)]
data021718_2 = data[['微博中文内容']][((data['date']=='2020-02-17')|(data['date']=='2020-02-18'))&(data['情感倾向']==1)]
data021718_0.to_csv('data021718_0.txt', index=False, header=False, sep='\t')
data021718_1.to_csv('data021718_1.txt', index=False, header=False, sep='\t')
data021718_2.to_csv('data021718_2.txt', index=False, header=False, sep='\t')



