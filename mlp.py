import pandas
import numpy as np

from sklearn.model_selection import train_test_split

import pandas as pd
 
from sklearn.preprocessing import LabelEncoder
label_encoder_teste = LabelEncoder()

hit_rate_train=[]

error_rate_train=[]
df=pd.read_csv('features.csv')
X = df.iloc[:, 0:25].values

y = df.iloc[:, 25].values


label_encoder_workclass = LabelEncoder()

y = label_encoder_workclass.fit_transform(y)




from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

y_labed = np.zeros((len(y), 4))

for i in range(len(y)):
    y_labed[i, y[i]] = 1


X = (X-np.mean(X))/(np.std(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labed, test_size=0.3)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)



feature_set =X_train 

labels = y_train 

one_hot_labels = y_train





def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 20
output_labels = 4

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 10e-3
error_cost = []
def getHitRate(y_predict,y_true):
    hits = 0
    for sample in range(0,len( y_predict)):
        is_hit=decison(y_predict[sample])==np.array(y_true[sample]).tolist()
        
        
        if(is_hit):
            hits=hits+1
    return 100*hits/len(y_true)
def decison(g):
    classification_smaple=[]
    g=np.array(g)
  
    for i in range(0,len(g)):
        if(g[i]>0.5):
            classification_smaple.append(1)
        else: classification_smaple.append(0);
         
    
    return classification_smaple;

for epoch in range(400):
############# feedforward

    # Phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = sigmoid(zo)

########## Back Propagation

########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

 
    loss = np.sum((ao-one_hot_labels)**2  )
    hit_rate_train.append(getHitRate(ao,one_hot_labels))
    error_rate_train.append(100-getHitRate(ao,one_hot_labels))
    error_cost.append(loss)
 




plt.figure() 
plt.plot(error_rate_train,label='Error rate(%)')
plt.plot(hit_rate_train,label='Hit rate(%)')
plt.xlabel('Epoch')
plt.legend()


   # Phase 1
zh = np.dot(X_test, wh) + bh
ah = sigmoid(zh)

    # Phase 2
zo = np.dot(ah, wo) + bo
ao = sigmoid(zo)



print(getHitRate(ao,y_test))

plt.show()
