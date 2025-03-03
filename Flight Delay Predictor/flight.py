#Initialising our Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#Import the CSV file
df=pd.read_csv('Flight_delay.csv')
df.head()

df=[['DayofWeek','Date','Deptime','Airline','Origin','Dest','CarrierDelay']]

#Check for missing data
df['Date']=pd.to_datetime(df['Date'],dayfirst=True)

#Create month and day feature
df['month']=df['Date'].dt.month
df['day']=df['Date'].dt.day

#Drop the date now
df=df.drop(columns=['Date'])

#Identify Categorical Variables
categories=df.select_dtypes(include=['object']).columns

#Encoding via Dummy Variables
df_encoded=pd.get_dummies(df,drop_first=True)

#Prepare Target Values
df_encoded['is_delayed_60+']=np.where(df_encoded['CarrierDelay']>60,1,0)

#Define Features and Target Variable
X=df_encoded.drop(columns=['is_delayed_60+','CarrierDelay'])
Y=df_encoded['is_delayed_60+']

#Split and test the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

#Training the data into one concatenated dataframe
train_set=pd.concat([X_train,Y_train],axis=1)

#Reversing one hot encoding 
for category in categories:
    one_hot_columns=[col for col in train_set.columns if col.startswith(f'{category}_')]
    train_set[category]=train_set[one_hot_columns].idxmax(axis=1)
    train_set=train_set.drop(columns=one_hot_columns)
    train_set[category]=train_set[category].str.replace(f'{category}_','')

#Check the Distribution of the Target Variable
train_set['is_delayed_60+'].value_counts()
train_set['is_delayed_60+'].mean()

#Delays by Airline
train_set.groupby('Airline')['is_delayed_60+'].mean().sort_values(ascending=False).round(3)*100

#Delays by Days of the Week
DayOfWeek_pct_delayed=train_set.groupby('DayOfWeek')['is_delayed_60+'].mean().round(3)*100

#Delays by Airport of Origin
pct_delay_by_origin=train_set.groupby('Origin')['is_delayed_60+'].mean().sort_values(ascending=False).round(3)*100
pct_delay_by_origin.head(20)

#Plot Histogram
plt.figure(figsize(10,6))
plt.hist(pct_delay_by_origin.values,bins=25,color='blue',edgecolor='black')

#Add Labels and Title
plt.title("Distribution of 60+ Minute Delays by Origins",fontsize=14)
plt.xlabel("Percentage of 60+ Minute Delays (%)",fontsize=12)
plt.ylabel("Frequency",fontsize=12)

#Show the Plot
plt.show()

#Model Training
xgb_model=xgb.XGBClassifier(random_state=0,eval_metric='logloss')
xgb_model.fit(X_train,Y_train)
y_pred=xgb_model.predict(X_test)

#Evaluate the Model
print("XGBoost Classifier (Baseline):")
print(f"Accuracy:{accuracy_score(Y_test,y_pred):.4f}")

#Confusion Matrix
cm=confusion_matrix(Y_test,y_pred)

#Predict probabilities for test set (to calculate AUC)
y_pred_proba=xgb_model.predict_proba(X_test)[:, 1] #We need probabilities for the positive class

#Calculate the AUC Score
auc_score=roc_auc_score(Y_test,y_pred_proba)
print(f"Auc Score:{auc_score:.4f}")

#Plot the ROC Curve
fpr,tpr,thresholds=roc_curve(Y_test,y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'AUC={auc_score:.4f}')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Cross Validated Grid Search
param_grid={'learning rate':[0.01,0.2],'max_depth':[3,5,7],'n_estimators':[100,250],'subsample':[0.6,1.0]}
xgb_model=xgb.XGBClassifier(random_state=0,eval_metric='logloss')

#Set up Grid Search CV
grid_search=GridSearchCV(estimator=xgb_model,param_grid=param_grid,cv=3,scoring='roc_auc',verbose=1,n_jobs=-1)

#Fit the Grid Search Model
grid_search.fit(X_train,Y_train)
