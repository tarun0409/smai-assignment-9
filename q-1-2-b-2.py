import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

class KMeans:
    centroids = None
    n_clusters = 5
    def fit(self, X, n_clusters):
        
        self.centroids = dict()
        self.n_clusters = n_clusters
        random.seed(1000)

        for i in range(0,n_clusters):
            curr_index = random.randint(0,len(X)-1)
            self.centroids[i] = X.iloc[curr_index].values

        clusters = [0]*X.shape[0]
        update_occurred = True 
        while update_occurred:
            update_occurred = False
            cluster_sum = dict()
            cluster_cardinality = dict()
            for i in range(0,n_clusters):
                cluster_sum[i] = np.zeros((1,X.shape[1]))
                cluster_cardinality[i] = 0
            index = 0
            for rand_index,row in X.iterrows():
                x = row.values
                best_cluster = -1
                best_distance = -1
                for i in range(0,n_clusters):
                    c = self.centroids[i]
                    dist = np.sum(np.square(np.subtract(c,x)))
                    if best_distance < 0 or dist < best_distance:
                        best_distance = dist
                        best_cluster = i
                if best_cluster != clusters[index]:
                    clusters[index] = best_cluster
                    update_occurred = True
                curr_cluster = clusters[index]
                cluster_sum[curr_cluster] += row.values.reshape(1,X.shape[1])
                cluster_cardinality[curr_cluster] += 1
                index += 1
            for i in range(0,n_clusters):
                self.centroids[i] = cluster_sum[i]/cluster_cardinality[i]

    def predict(self, X):
        y_predict = list()
        for index,row in X.iterrows():
            x = row.values
            best_cluster = -1
            best_distance = -1
            for i in range(0,self.n_clusters):
                c = self.centroids[i]
                dist = np.sum(np.square(np.subtract(c,x)))
                if best_distance < 0 or dist < best_distance:
                    best_distance = dist
                    best_cluster = i
            y_predict.append(best_cluster)
        return y_predict
    
    def compute_purity(self, y_predict, y_actual):
        correct = 0
        for i in range(0,len(y_actual)):
            if y_actual[i] == y_predict[i]:
                correct += 1
        purity = float(correct)/float(len(y_actual))
        return purity

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

data = pd.read_csv("compressed_intrusion_data_b_2.csv",header=None)
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
    test_size=0.2,
    random_state=0)

kms = KMeans()
kms.fit(X_train, 5)
y_pred_tr_kms = list(kms.predict(X_train))
y_pred_ts_kms = list(kms.predict(X_test))

train_purity = compute_purity(y_train, y_pred_tr_kms, y_train, y_pred_tr_kms, 'xAttack')
test_purity = compute_purity(y_train, y_pred_tr_kms, y_test, y_pred_ts_kms, 'xAttack')
print '******************** K-Means Clustering result ********************************'
print 'Train data set purity : '+str(round(train_purity*100,2))+'%'
print 'Test data set purity : '+str(round(test_purity*100,2))+'%'
print '************************************************************************'



