###################################################################
#                          Assignment- 2                          #
#                            Group- 21                            #
#                   Sharannya Ghosh (20CS10054)                   #
#                     Aritra Mitra  (20CS30006)                   #
#                            Problem- 2                           #
###################################################################

import numpy as np
from sklearn import svm, neural_network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
np.random.seed(16)

# 1.

# to encode categorical values
def encoding(string):
    if(string=='Iris-setosa'):
        return 0
    if(string=='Iris-versicolor'):
        return 1
    return 2

data = pd.read_csv("bezdekIris.data",header=None)
for i in data.columns:
    if(data[i].dtype==np.float64):
        data[i] = (data[i]-data[i].mean())/data[i].std()    #Standard Scalar Normalization
    else:
        data[i] = data[i].apply(encoding)   #encoding strings

def setosa_bin(data_value):         #for binary classification of a single flower (IS{0} / IS NOT{1})
    if data_value==0:
        return 0
    else:
        return 1
def versicolor_bin(data_value):
    if data_value==1:
        return 0
    else:
        return 1
def virginica_bin(data_value):
    if data_value==2:
        return 0
    else:
        return 1

#List of functions used for SVM application
funclist=[setosa_bin,versicolor_bin,virginica_bin]
#Partitioning Training and Testing Datasets
train=data.sample(frac=0.8)
test=data.drop(train.index)
test=test.reset_index(drop=True)
train=train.reset_index(drop=True)
train_X=train[(x for x in range(4))]
train_Y=train[4]
test_X=test[(x for x in range(4))]
test_Y=test[4]

#2.

predictions_SVMrad=[]
predictions_SVMquad=[]
for i in range(3):
    #Applying SVM to classify one type of flower from the others
    train_Y_t = train_Y.apply(funclist[i])  #Applying appropriate functions for classification purposes
    test_Y_t = test_Y.apply(funclist[i])
    '''
        Linear Kernel
    '''
    SVM = svm.SVC(kernel='linear')
    SVM.fit(train_X,train_Y_t)
    predictions_SVM = SVM.predict(test_X)
    # print(predictions_SVM)
    print("SVM Accuracy Score for binary classification of "+(funclist[i].__name__)[:-4]+" from rest using linear kernel -> ",accuracy_score(predictions_SVM, test_Y_t)*100)
    '''
        Radial Basis Function Kernel
    '''
    SVMrad= svm.SVC(kernel='rbf')
    SVMrad.fit(train_X,train_Y_t)
    predictions_SVMrad.append(SVMrad.predict(test_X))
    # print(predictions_SVM)
    print("SVM Accuracy Score for binary classification of "+(funclist[i].__name__)[:-4]+" from rest using radial basis function kernel -> ",accuracy_score(predictions_SVMrad[i], test_Y_t)*100)
    '''
        Quadratic Kernel
    '''
    SVMquad = svm.SVC(kernel='poly',degree=2)
    SVMquad.fit(train_X,train_Y_t)
    predictions_SVMquad.append(SVMquad.predict(test_X))
    # print(predictions_SVM)
    print("SVM Accuracy Score for binary classification of "+(funclist[i].__name__)[:-4]+" from rest using quadratic kernel -> ",accuracy_score(predictions_SVMquad[i], test_Y_t)*100)

#3.

size_a = (16)   #neural network hidden layers for a. part
size_b = (256, 16)  #neural network hidden layers for b. part
MLP_a = neural_network.MLPClassifier(hidden_layer_sizes=size_a,batch_size=32,learning_rate='constant',learning_rate_init=0.001)
MLP_a.fit(train_X,train_Y)
predictions_MLP_a = MLP_a.predict(test_X)
accuracy_a = accuracy_score(predictions_MLP_a,test_Y)*100
print("MLP Classifier accuracy score for Part a. -> ",accuracy_score(predictions_MLP_a,test_Y)*100)
MLP_b = neural_network.MLPClassifier(hidden_layer_sizes=size_b,batch_size=32,learning_rate='constant',learning_rate_init=0.001)
MLP_b.fit(train_X,train_Y)
predictions_MLP_b = MLP_b.predict(test_X)
accuracy_b = accuracy_score(predictions_MLP_b,test_Y)*100
print("MLP Classifier accuracy score for Part b. -> ",accuracy_score(predictions_MLP_b,test_Y)*100)

#4.

optimal = 'a' if accuracy_a>accuracy_b else 'b'     #finding the optimal model
learning_rate=0.1
level = np.zeros((5,2))     #matrix to store the outcomes
selected_MLP_parameter = globals()['size_'+str(optimal)]    #finding the parameters of that model
for i in range(5):
    #print(selected_MLP_parameter)
    MLP = neural_network.MLPClassifier(hidden_layer_sizes=selected_MLP_parameter,batch_size=32,learning_rate='constant',learning_rate_init=learning_rate)
    learning_rate *= 0.1
    MLP.fit(train_X, train_Y)
    level[i][0]=learning_rate
    level[i][1]=accuracy_score((MLP.predict(test_X)),test_Y)*100
    df = pd.DataFrame({'Learning Rate': level[:,0], 'Accuracy': level[:,1]})
#print(level)
sns.set(rc={"figure.figsize":(10, 7)})
plot=sns.lineplot(data=df, x='Learning Rate', y='Accuracy')
plot=sns.scatterplot(data=df, x='Learning Rate', y='Accuracy')
plot.set(xscale='log')  #plotting the data in logarithmic scale of learning rate for better visualization
plt.savefig("q2_graph.png")

#5.

def removefeature(df,dftest):
    newtrain=None
    newtest=None
    feature=-1
    lr=0.001
    if(len(df.columns)>1):
        MLPold=neural_network.MLPClassifier(hidden_layer_sizes=selected_MLP_parameter,batch_size=32,learning_rate='constant',learning_rate_init=lr)
        MLPold.fit(df,train_Y)
        predictions=MLPold.predict(dftest)
        acc=accuracy_score(predictions,test_Y)*100
        for column in df.columns:
            newtrain=df.drop(column,axis=1)
            newtest=dftest.drop(column,axis=1)
            MLPnew= neural_network.MLPClassifier(hidden_layer_sizes=selected_MLP_parameter,batch_size=32,learning_rate='constant',learning_rate_init=lr)
            MLPnew.fit(newtrain, train_Y)
            newpredict=MLPnew.predict(newtest)
            newacc=accuracy_score(newpredict,test_Y)*100
            if(newacc>acc):
                acc=newacc
                feature=column
    return feature

reducedX=train_X
reducedXtest=test_X
column=-1
while(True):
      column=removefeature(reducedX,reducedXtest)
      # print(column)
      if(column<0):
        print("Features: ")
        for c in reducedX.columns:
            if(c == 0):
                print('Sepal Width,',end=' ')
            if(c == 1):
                print('Sepal Length,',end=' ')
            if(c == 2):
                print('Petal Width,',end=' ')
            if(c == 3):
                print('Petal Length,',end=' ')
        print('')
        break
      else:
        reducedX.drop(column,axis=1,inplace=True)
        reducedXtest.drop(column,axis=1,inplace=True)

#6.

predmlp=globals()['predictions_MLP_'+str(optimal)]
pred=[[],[],[]]
for i in range(3):
    binmlp=pd.Series(predmlp).apply(funclist[i])
    count0=0
    count1=1
    for j in range(0,test_Y.shape[0]):
        count0=(1-binmlp[j])+(1-predictions_SVMrad[i][j])+(1-predictions_SVMquad[i][j])
        count1=binmlp[j]+predictions_SVMrad[i][j]+predictions_SVMquad[i][j]
        if(count0>=count1):
            pred[i].append(0)
        else:
            pred[i].append(1)
    test_Y_t = test_Y.apply(funclist[i])     
    accscore=accuracy_score(pred[i],test_Y_t)*100
    print("Accuracy Score for Ensemble Learning of "+(funclist[i].__name__)[:-4]+" -> ",accscore)

#6. (Alternative)

for i in range(3):
    test_Y_t = test_Y.apply(funclist[i])
    train_Y_t = train_Y.apply(funclist[i])
    MLP = neural_network.MLPClassifier(hidden_layer_sizes=selected_MLP_parameter, batch_size=32, learning_rate='constant', learning_rate_init=0.001)
    MLP.fit(train_X,train_Y_t)
    MLP_pred = MLP.predict(test_X)
    pred=[[],[],[]]
    for j in range(0,test_Y.shape[0]):
        count0=(1-binmlp[j])+(1-predictions_SVMrad[i][j])+(1-predictions_SVMquad[i][j])
        count1=binmlp[j]+predictions_SVMrad[i][j]+predictions_SVMquad[i][j]
        if(count0>count1):
            pred[i].append(0)
        else:
            pred[i].append(1)
    accscore=accuracy_score(pred[i],test_Y_t)*100
    print("Accuracy Score for Ensemble Learning of "+(funclist[i].__name__)[:-4]+" using alternative method-> ",accscore)