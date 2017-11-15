from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
import numpy as np


def Sigmas_fuc(Cluster_centers):
    n_clusters = np.shape(Cluster_centers)[0]
    dimensions = np.shape(Cluster_centers)[1]
    Distances = np.ndarray(shape=(n_clusters,1),dtype = 'float')
    Sigmas = np.ndarray(shape=(n_clusters,1),dtype = 'float')
    for I in range(n_clusters):
        for J in range(n_clusters):
            if I == J:
                Distances[J] = 0
            else:
                Distances[J] = np.linalg.norm(Cluster_centers[I]-Cluster_centers[J])
        max_dist = Distances.max()
        Sigmas[I] = max_dist/pow(2*n_clusters,0.5)
    return Sigmas

def Gaussian_fuc(X,center,sigma):
    gauss = np.exp(-pow(np.linalg.norm(X-center),2)/(2*pow(sigma,2)))
    return gauss

def Green_fuc(Xs, Cluster_centers,Sigmas):
    n_clusters = np.shape(Cluster_centers)[0]
    n_datas = np.shape(Xs)[0]
    Green = np.ndarray(shape=(n_datas,n_clusters),dtype = 'float')
    for I in range(n_datas):
        for J in range(n_clusters):
            Green[I,J] = Gaussian_fuc(Xs[I],Cluster_centers[J],Sigmas[J])
    return Green

def W_fuc(Green,d,regular):
    I = np.eye(np.shape(Green)[0],np.shape(Green)[1])
    return np.dot(np.linalg.pinv(Green+regular*I),d)

#---------------------------------------------------------------------------------#


def data_split(Dataset):
    datas = Dataset["data"]
    targets = Dataset["target"]
    encoder = OneHotEncoder()
    encoder.fit(np.unique(targets)[:,None])
    targets = encoder.transform(targets[:,None]).toarray()
    permutation = np.random.permutation(np.shape(targets)[0])
    datas = datas[permutation,:]
    targets = targets[permutation]
    # tra_datas,vali_datas,tra_targets,vali_targets = train_test_split(datas,targets,test_size = 0.5)
    vali_number = np.shape(targets)[0]//5
    dataset = []
    labelset = []
    training_set = []
    vali_set = []
    training_labels = []
    vali_lables = []
    for I in range(5):
        dataset.append(list(datas[I*vali_number:(I+1)*vali_number,:]))
        labelset.append(list(targets[I*vali_number:(I+1)*vali_number]))
    for I in range(5):
        tra_minibatch = []
        label_minibatch = []
        for J in range(5):
            if J != I:
                tra_minibatch += dataset[J]
                label_minibatch += labelset[J]
        training_set.append(tra_minibatch)
        training_labels.append(label_minibatch)
        tra_minibatch = []
        label_minibatch = []
        vali_set.append(dataset[I])
        vali_lables.append(labelset[I])
    return np.array(training_set),np.array(training_labels),np.array(vali_set),np.array(vali_lables)

def normalization(tra_datas,vali_datas):
    ss = MinMaxScaler(feature_range=[-1,1],copy=False)
    tra_datas = ss.fit_transform(tra_datas)
    vali_datas = ss.transform(vali_datas)
    return tra_datas,vali_datas

def find_centers(n_neurons):
    n_neurons = n_neurons
    kmeans = KMeans(n_clusters = n_neurons)
    kmeans.fit(tra_datas)
    return kmeans

def train_network(tra_datas,tra_targets,regular):
    kmeans = find_centers(4)
    Sigmas = Sigmas_fuc(kmeans.cluster_centers_)
    Green = Green_fuc(tra_datas,kmeans.cluster_centers_,Sigmas)
    W = W_fuc(Green,tra_targets,regular)
    return W,kmeans,Sigmas

def predict(vali_datas,W,kmeans,Sigmas):
    prediction = np.dot(Green_fuc(vali_datas,kmeans.cluster_centers_,Sigmas),W)
    return prediction

def accuracy(predict,vali_targets):
    err = predict.argmax(1)-vali_targets.argmax(1)
    accu = err[err==0].shape[0]/vali_targets.shape[0]
    return accu




iris = datasets.load_iris()

regulars = np.arange(0,10,0.1)
accus = np.ndarray(shape=(len(regulars)), dtype = 'float')
for J in range(len(regulars)):
    tra_set,tra_labels,vali_set,vali_labels = data_split(iris)
    accu = np.ndarray(shape = (5,1), dtype = 'float')
    for I in range(5):

        tra_datas,vali_datas = normalization(tra_set[I],vali_set[I])
        W,kmeans,Sigmas = train_network(tra_datas,tra_labels[I],regulars[J])
        prediction = predict(vali_datas,W,kmeans,Sigmas)
        accu[I] = accuracy(prediction,vali_labels[I])
    accus[J] = sum(accu)/5
plt.figure()
plt.plot(regulars,accus)
plt.show()
