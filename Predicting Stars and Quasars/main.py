#All Imports
import os
#import Data Manipulation Libraries
import numpy as np
import pandas as pd
#import Libraries for Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Import Libraries for Model Building
import tensorflow as tf
from tensorflow import keras
#Import Libraries for Data Analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#Import Libraries for Model Accuracy
from sklearn.metrics import accuracy_score
#Import Libraries for Classification and Regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
#Exploratory Analysis
data=pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv')
data.head
data.shape
data.drop(['objid','specobjid'],axis=1,inplace=True)
data.head(10)
data.shape
data.describe()
data.info()
#Convert Category Data to Numeric Data
le=LabelEncoder().fit(data['class'])
data['class']=le.transform(data['class'])
#Final Dataset
data.head(10)
data.info()
#Training and splitting the data
X=data.drop('class',axis=1)
Y=data['class']
#Data Scaling
scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
X=scaler.fit_transform(X)
X[:20]
#Train, test, Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,size=0.3,random_state=128)
#Density Distribution Plots
sns.countplot(x=data['class'])
#Using Pairplots to establish and understand interdependency of train features
sns.pairplot(data[['u','g','r','i','class']])
#Decision Tree Classifier
dtClassifier=DecisionTreeClassifier(max_leaf_nodes=15,max_depth=3)
#Linear Classfier=Logistic Regression
LRClassifier=LogisticRegression()
#Nearest Neighbour Classifier
NeNeClassifier=KNeighborsClassifier(n_neighbors=3)
#Fitting the Models to the Dataset
dtClassifier.fit(X_train,Y_train)
LRClassifier.fit(X_train,Y_train)
NeNeClassifier.fit(X_train,Y_train)
#Getting the prediction set of the models
y_preds=dtClassifier.predict(X_test)
y_predsLR=LRClassifier.predict(X_test)
y_predsNeNe=NeNeClassifier.predict(X_test)
#Displaying the predictions as output by the classifier
print(y_preds[:10],'\n',y_test[:10])
print("\n***********************************************************")
print(y_predsLR[:10],'\n',y_test[:10])
print("\n***********************************************************")
print(y_predsNeNe[:10],'\n',y_test[:10])
#Classification Report
target_names=['0','1','2']
print('\033[lm Decision Tree -\n]\033[0m',classification_report(y_preds,Y_test,target_names=target_names)+'\n')
print('\033[lm Linear Regression -\n]\033[0m',classification_report(y_predsLR,Y_test,target_names=target_names)+'\n')
print('\033[lm KNN Classifier -\n]\033[0m',classification_report(y_predsNene,Y_test,target_names=target_names)+'\n')
