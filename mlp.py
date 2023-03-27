import pandas
import numpy as np

from sklearn.model_selection import train_test_split

import pandas as pd
df=pd.read_csv('file.csv')
df=df.drop(['Id'],axis=1)
from sklearn.preprocessing import LabelEncoder
label_encoder_teste = LabelEncoder()

X = df.iloc[:, 0:4].values

y = df.iloc[:, 4].values


label_encoder_workclass = LabelEncoder()

y = label_encoder_workclass.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

def tanh(x):
  return np.tanh(x)

LR = 0.1   

I_dim = 4
H_dim = 4

epoch_count = 800

 
weights_ItoH = np.random.uniform(-0.5, 0.5, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-0.5, 0.5, H_dim)

preActivation_H = np.zeros(H_dim)
postActivation_H = np.zeros(H_dim)

training_data =X_train
target_output = y_train
training_data = np.asarray(training_data)
training_count = len(training_data[:,0])

validation_data = X_test
validation_output = y_test
validation_data = np.asarray(validation_data)
validation_count = len(validation_data[:,0])

#####################
#training
#####################
for epoch in range(epoch_count):

    for sample in range(training_count):
        for node in range(H_dim):
            preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])
            
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)
        
        FE = postActivation_O - target_output[sample]
        
        for H_node in range(H_dim):
            S_error = FE * logistic_deriv(preActivation_O)
            gradient_HtoO = S_error * postActivation_H[H_node]
                       
            for I_node in range(I_dim):
                input_value = training_data[sample, I_node]
                gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value
                
                weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
                
            weights_HtoO[H_node] -= LR * gradient_HtoO

#####################
#validation
#####################            
correct_classification_count = 0
for sample in range(validation_count):
    for node in range(H_dim):
        preActivation_H[node] = np.dot(validation_data[sample,:], weights_ItoH[:, node])
        postActivation_H[node] = logistic(preActivation_H[node])
            
    preActivation_O = np.dot(postActivation_H, weights_HtoO)
    postActivation_O = logistic(preActivation_O)
        
    if postActivation_O >0:
        output = 1
    else:
        output = 0    
        
    if output == validation_output[sample]:
        correct_classification_count += 1

print('Accuracy:')
print(correct_classification_count*100/validation_count)