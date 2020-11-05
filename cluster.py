# 对于高频词用BERT词向量表示然后聚类
from bert_serving.client import BertClient
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib
bc = BertClient()
km = KMeans(n_clusters=3)
pca = PCA(n_components=2)
data = pd.read_csv(r"D:\workspace\ML\nlp\comment\2020-05-25-fc.csv")
corpus = data.iloc[0:120,1].tolist()

vectors = bc.encode(corpus)
vectors_ = pca.fit_transform(vectors)   #降维到二维
y_ = km.fit_predict(vectors_)       #聚类
plt.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['axes.unicode_minus']=False
plt.scatter(vectors_[:,0],vectors_[:, 1],c=y_)   #将点画在图上
for i in range(len(corpus)):    #给每个点进行标注
    plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
plt.show()