import pandas as pd
from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

import numpy as np
import warnings
from keras.callbacks import History
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('crop_production.csv', encoding='utf-8')
df = df[df['State_Name'] == "Andhra Pradesh"]
df['Yield'] = df['Production']/df['Area']


df = df[df['Crop_Year']>=2004]
df.head()
df = df.join(pd.get_dummies(df['District_Name']))
df = df.join(pd.get_dummies(df['Season']))
df = df.join(pd.get_dummies(df['Crop']))
#df = df.join(pd.get_dummies(df['Crop_Year']))
df = df.join(pd.get_dummies(df['State_Name']))

df=df.drop('District_Name', axis=1)
df = df.drop('Season',axis=1)
df = df.drop('Crop',axis=1)
df = df.drop('Crop_Year', axis=1)
df = df.drop('Production', axis=1)
df = df.drop('State_Name', axis=1)

C_mat = df.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8 ,square=True)
plt.show()



x = df[['Area']].values.astype(float)


# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)
x_scaled

df['Area'] = x_scaled

df = df.fillna(df.mean())


b = df['Yield']
a = df.drop('Yield', axis = 1)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.3, random_state = 42)
a_test.head()


def compare_models(model):
    model_name = model.__class__.__name__
    fit=model.fit(a_train,b_train)
    y_pred=fit.predict(a_test)
    r2=r2_score(b_test,y_pred)
    return([model_name,r2])

NN_model=Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = a_train.shape[1], activation='relu'))

# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()



models = [
     GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=0),
     RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0),
    svm.SVR(),
   DecisionTreeRegressor(),
   XGBRegressor(),
]

model_train=list(map(compare_models,models))
print(*model_train, sep = "\n")


history = History()
History=NN_model.fit(a_train, b_train, epochs=50, batch_size=500, validation_split = 0.2, callbacks=[history])
gb=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
gb.fit(a_train,b_train)
Pkl_Filename = "Pickle_RL_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(gb, file)
predd=NN_model.predict(a_test)
print(r2_score(b_test,predd))

print(a_test.head())