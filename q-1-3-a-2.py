import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import mixture
np.random.seed(943)

def compute_purity(y_train, y_train_predict, y_actual, y_predict, y_label):
    
    cluster_label_map = dict()
    unique, counts = np.unique(y_train_predict, return_counts=True)
    cluster_dict = dict(zip(unique, counts))
    class_dict = dict()
    y_list = y_train[y_label].tolist()
    for i in y_train[y_label].unique():
        class_dict[i] = y_list.count(i)
    while cluster_dict:
        cluster = max(cluster_dict,key=cluster_dict.get)
        clas = max(class_dict,key=class_dict.get)
        cluster_label_map[cluster] = clas
        del cluster_dict[cluster]
        del class_dict[clas]
    y_pred = map(lambda x : cluster_label_map[x], y_predict)
    y_act = y_actual[y_label].tolist()
    
    correct = 0
    for i in range(0,len(y_act)):
        if y_act[i] == y_pred[i]:
            correct += 1
    purity = float(correct)/float(len(y_act))
    return purity

data = pd.read_csv("compressed_intrusion_data_a_2.csv",header=None)
heading = list()
for i in range(1,15):
    heading.append('A'+str(i))
heading.append('xAttack')
data.columns = heading

cols = list()
for i in range(1,15):
    cols.append('A'+str(i))
X_train, X_test, y_train, y_test = train_test_split(
    data[cols],
    data[['xAttack']],
    test_size=0.3,
    random_state=0)

gmm = mixture.GaussianMixture(n_components=5)
gmm.fit(X_train)
y_pred_tr_gmm = list(gmm.predict(X_train))
y_pred_ts_gmm = list(gmm.predict(X_test))

train_purity = compute_purity(y_train, y_pred_tr_gmm, y_train, y_pred_tr_gmm, 'xAttack')
test_purity = compute_purity(y_train, y_pred_tr_gmm, y_test, y_pred_ts_gmm, 'xAttack')
print '******************** Gaussian mixture models Clustering result ********************************'
print 'Train data set purity : '+str(round(train_purity*100,2))+'%'
print 'Test data set purity : '+str(round(test_purity*100,2))+'%'
print '************************************************************************'

