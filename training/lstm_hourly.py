import keras
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20, 15)
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import copy
import random 

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs1', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)



neurons = 75
dropout = 0.2
layers = 6
epochs = 8
batch_size = 8

training_size = 6000 # datapoints for training
extra_params = 3 # 1 = temp , 2 = temp,weekday, 3 = temp,weekday,workday


timestep = 48 # Number of previous hours we nfeed to the neural net
total_hours_predict = 252# Total hours to predict
hours_per_prediction = 36 # Hours to predict per step

df= pd.read_csv('data/hourly/YOURFILE.csv')
df = df.drop(['Month','Weekday'],axis=1)
df_length = df.shape[0]

training_set = df.iloc[0:training_size,:]

training_set = training_set.iloc[:,1:(2 + extra_params)].values

#Feature scaling

sc = MinMaxScaler(feature_range = (0,1))  #StandardScaler for standatization and MinMaxScaler forn normalization 
#sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []


for i in range(timestep, len(training_set)):
    X_train.append(training_set_scaled[i - timestep:i ,0:1 + extra_params])
    y_train.append(training_set_scaled[i ,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1 + extra_params))


regressor = Sequential()

regressor.add(LSTM(units = neurons, return_sequences = True,input_shape = (X_train.shape[1], 1 + extra_params )))
regressor.add(Dropout(dropout))
BatchNormalization(axis=-1)
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(dropout))
BatchNormalization(axis=-1)
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(dropout))
BatchNormalization(axis=-1)
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(dropout))
BatchNormalization(axis=-1)
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(dropout))

#regressor.add(CuDNNLSTM(units = neurons, return_sequences = True))
#regressor.add(Dropout(dropout))
#
#regressor.add(CuDNNLSTM(units = neurons, return_sequences = True))
#regressor.add(Dropout(dropout))

regressor.add(LSTM(units = neurons))
regressor.add(Activation('linear'))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = "mae")

early_stop = EarlyStopping(monitor='loss',min_delta=0,patience=0,
                              verbose=0, mode='auto')

loss =regressor.fit(X_train, y_train, epochs = epochs,batch_size = batch_size,
                   callbacks=[early_stop])
#from keras.models import load_model
#regressor = load_model("lstm150.h5")
prediction_starting_point = random.randint(0,200)
start_date = df.iloc[training_size  + prediction_starting_point,0]
end_date = df.iloc[training_size  + prediction_starting_point + total_hours_predict,0]
total_energy_usage = df.iloc[:,1:2 + extra_params ]
test_set = df.iloc[training_size + prediction_starting_point : training_size + prediction_starting_point + total_hours_predict,:]
test_set = test_set.iloc[:,1:2 + extra_params].values

inputs = total_energy_usage[(training_size + prediction_starting_point) - timestep : (training_size + prediction_starting_point) + total_hours_predict ].values
inputs = sc.transform(inputs)

X_test = []
for i in range(0,total_hours_predict):
    X_test.append(inputs[i :  timestep  + i, 0 : 1 + extra_params])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1 + extra_params))



predicted_energy_usage = []
for i in range (int(total_hours_predict/hours_per_prediction)):
    curr_test = copy.deepcopy(inputs[i * hours_per_prediction : i * hours_per_prediction + timestep + hours_per_prediction])
    for y in range(0,hours_per_prediction):
        test = np.expand_dims(curr_test[y:timestep+y,:],axis=0)
        predicted = regressor.predict(test)[0][0]
        curr_test[timestep + y,0] = predicted
        predicted_energy_usage.append(predicted.tolist())

predicted_energy_usage = np.expand_dims(np.asarray(predicted_energy_usage),axis=1)        

real_energy_usage = test_set[:,0]
predicted_energy_usage_tbl = np.zeros(shape=(len(predicted_energy_usage), 1 + extra_params) )
predicted_energy_usage_tbl[:,0] = predicted_energy_usage[:,0]
predicted_energy_usage = sc.inverse_transform(predicted_energy_usage_tbl)[:,0]
#



def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse = sqrt(mean_squared_error(real_energy_usage,predicted_energy_usage ))
mae =  mean_absolute_error(real_energy_usage,predicted_energy_usage )
mape =   mean_absolute_percentage_error(real_energy_usage,predicted_energy_usage )
time = df.iloc[training_size + prediction_starting_point : training_size + total_hours_predict ,:1].values.tolist()

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
major_ticks = np.arange(0, total_hours_predict, hours_per_prediction) 
ax.set_xticks(major_ticks)
ax.grid()
ax.plot(real_energy_usage, color = 'red',  label = 'real energy usage',linestyle='dotted')
ax.plot(test_set[:,1], color = 'green', label = 'temperature', linestyle='dashed',alpha=0.2)

for i  in range(0,total_hours_predict,hours_per_prediction):
    x = (np.arange(i,i + hours_per_prediction )).tolist()
    y = predicted_energy_usage[i :i + hours_per_prediction]
    plt.plot(x,y , label = " ")

ax.set_title('Energy Usage prediction LSTM')
ax.set_xlabel("Time(hrs)")
ax.set_ylabel('kWh / C')
ax.text(1,-7,"Layers: " + str(layers) + " | epochs: " + str(epochs) + " | neurons:" + str(neurons) + 
        " | batch_size: " +  str(batch_size) + " | dropout: " + str(dropout) + " | timestep: " + str(timestep) + " | " +
        "params: " + str(extra_params), fontsize=10)
ax.text(0,-5,"RMSE: " + str(round(rmse, 2)) + " | mae: " + str(round(mae, 2)) + " | mape: " + str(round(mape, 2)) + "%" + " | "
        "total_hours_predict: " + str(total_hours_predict) + " | " +
        "hours_per_prediction: " + str(hours_per_prediction) )
ax.text(0,-3,"Start Date : " + start_date + " | " + "End Date: " + end_date)
ax.legend(loc="lower right")
plt.show()

#from keras.models import load_model
#regressor.save('lstm75_48hrs.h5') 