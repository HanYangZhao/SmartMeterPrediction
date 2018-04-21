from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/hourly/YOURFILE.csv')
data = data.drop(['Date and time','Month','Weekday'],axis=1)

sc = MinMaxScaler(feature_range = (0,1))  #StandardScaler for standatization and MinMaxScaler forn normalization 
#sc = StandardScaler()
#data = sc.fit_transform(data)

X = data.iloc[:, 1:4].values
y = data.iloc[:, 0].values
# dayEncoder = LabelEncoder()
# X[:, 1] = dayEncoder.fit_transform(X[:, 1])
# monthEncoder = LabelEncoder()
# X[:, 2] = monthEncoder.fit_transform(X[:, 2])

# onehotencoder = OneHotEncoder(categorical_features = 'all')

# day = X[:,1].reshape(-1,1) 
# day = onehotencoder.fit_transform(day).toarray()
# #remove first dummy variable to avoid trap
# day = day[:,1:]

# month = X[:,2].reshape(-1,1)
# month = onehotencoder.fit_transform(month).toarray()
# #remove first dummy variable to avoid trap
# month = month[:,1:]

# temp=X[:,0].reshape(-1,1)

# X = np.concatenate((day, month), axis=1)
# X = np.concatenate((X, temp), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

neurons = [150]
dimensions = np.shape(X)[1]
layers = 6
dropout = 0.2
def baseline_model(optimizer,units):
# Initialising the ANN
    model = Sequential()
     # Adding the input layer 
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=dimensions))
    for i in range(0, layers):
        #add hidden layers
        model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(dropout))
    #model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='relu'))
    
    # Compiling the ANN
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae', 'mape'])
    return model

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, verbose=1)

#hyperparameters tuning
parameters = {'batch_size': [100,200,400],
              'epochs': [50,100,200],
              'optimizer': ['adam'],
              'units': neurons,
              }

grid_search = GridSearchCV(estimator = estimator,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10)


grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)


real_energy_usage = []
temp = []
error = []
predicted = []

plt.rcParams['figure.figsize'] = (20, 15)
for i,day in enumerate(y_test):
    prediction = grid_search.predict((np.array([X_test[i]])))
    predicted.append(np.asscalar(prediction))
    # test.append(y_test[i])
    
    # temp.append(X_test[i][0])
    error.append(abs(y_test[i] - prediction))



real_energy_usage = y_test[:]
#real_energy_usage = test
#real_energy_usage = np.expand_dims(np.asarray(real_energy_usage),axis=1)       
#real_energy_usage_tbl = np.zeros(shape=(len(real_energy_usage), 1 + 2 ))
#real_energy_usage_tbl[:,0] = real_energy_usage[:,0]
#real_energy_usage  = sc.inverse_transform(real_energy_usage_tbl)[:,0]

temp = X_test[:,0]
temp = np.expand_dims(np.asarray(temp),axis=1)       
temp_tbl = np.zeros(shape=(len(temp), 1 + 2 ))
temp_tbl[:,0] = temp[:,0]
temp = sc.inverse_transform(temp_tbl)[:,0]


rmse = sqrt(mean_squared_error(real_energy_usage ,predicted))
avg_error =  sum(error) / float(len(error))

plt.title('Daily Electricity Usage vs Temp (multi-varaiate linear regression)')
plt.text(0,0,"Best Score" + str(round(best_accuracy,4)) + " | " 
         + "Rmse:" + str(round(rmse, 2)) +  " | "
         + "avg_error:" + str(round(avg_error, 2)) , fontsize=10)
plt.text(0,0.5,"units: " + str(neurons[0]) + " | " +  "layers: " + str(layers) +
         " | " + "input_dim" + str(dimensions) + " | "
         + "dropout: " + str(dropout) + " | " + str(best_parameters),fontsize=9)
plt.scatter(temp,predicted)
plt.scatter(temp,real_energy_usage)
plt.xlabel('Temperature (C)')
plt.ylabel('kWh')
plt.legend(['predicted', 'real'], loc='upper right')


#grid_search.best_estimator_.model.save('dnn_100.h5')
plt.show()
