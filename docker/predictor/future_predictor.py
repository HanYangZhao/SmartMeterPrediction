

from darksky import forecast
import numpy as np
import arrow
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt 
import copy


class Predictor:
    df_lstm = pd.read_csv('2017-07-01_2018-03-18.csv').iloc[:,1:5].values
    df_dnn = pd.read_csv('2017-07-01_2018-03-18.csv').drop(['Date and time','Month','Weekday'],axis=1).iloc[:,1:4].values

    def __init__(self):
        pass

    def ftoc(self, input):
        return round((input - 32) * 5.0/9.0, 1)
    
    def future_weather(self,api, lat, lon, hours, time):
        days = int(hours) // 24
        hour = int(hours) % 24
        data = []
        for i in range(days):
            t = time.shift(days=+i)
            t = dt.datetime(t.year,t.month, t.day, 0).isoformat()
            weather = forecast(api, lat, lon, time=t)
            data.extend([self.ftoc(hour.temperature) for hour in weather.hourly[:24]])
        if hour != 0:
            t = time.shift(days=+days)
            t = dt.datetime(t.year,t.month, t.day, 0).isoformat()
            weather = forecast(api, lat, lon, time=t)
            data.extend([self.ftoc(hour.temperature) for hour in weather.hourly[:hours]])
        return data

    def is_workday(self,now,i):
            weekday = (now + i // 24) % 7
            if(weekday > 4):
                return 0
            else:
                return 1

    def predict_future_dnn(self, predict_hours, model_file, weather):
        predicted = []
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(self.df_dnn)
        model = load_model(model_file)
        now = arrow.now().weekday()
        x = np.empty([predict_hours, 3])
        for i,temp in enumerate(weather):
            x[i, 0] = temp
            x[i, 1] = i % 24
            x[i, 2] = self.is_workday(now,i)
            #print(x[i,:])
        x = sc.transform(x)
        for i in range(predict_hours):
            predicted.append(np.asscalar(model.predict((np.array([x[i]])))))
        return predicted

    def predict_future_lstm(self, predict_hours, model_file, inputs, weather, inputs_steps):
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(self.df_lstm)
        model = load_model(model_file)
        is_workday = 1
        is_tmr_workday = 1
        weekday = arrow.now().weekday()
        if weekday > 4:
            is_workday = 0
        if weekday + 1 > 4:
            is_tmr_workday = 0
        x = np.empty([predict_hours, 4])
        for i,temp in enumerate(weather):
            x[i,0] = 0
            x[i, 1] = temp
            x[i, 2] = i % (len(weather))
            if i > 23:
                x[i, 3] = is_tmr_workday
            else:
                x[i, 3] = is_workday
        inputs = np.append(inputs, x, axis=0)
        
        predicted_energy_usage = []
        for i in range(predict_hours):
            input = sc.transform(inputs[i:i+inputs_steps, :])
            input = np.expand_dims(input, axis=0)
            predicted = model.predict(input)[0][0]
            predicted_energy_usage.append(predicted)
            inputs[inputs_steps+i, 0] = predicted
        predicted_energy_usage = np.expand_dims(np.asarray(predicted_energy_usage), axis=1)
        predicted_energy_usage_tbl = np.zeros(shape=(len(predicted_energy_usage), 4))
        predicted_energy_usage_tbl[:, 0] = predicted_energy_usage[:, 0]
        predicted_energy_usage = sc.inverse_transform(predicted_energy_usage_tbl)[:, 0]
        return predicted_energy_usage.tolist() 
    
    def plot_data(self, predictions, weather_data, model):
        
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.grid()
        major_ticks = np.arange(0, len(predictions[0]), 1) 
        ax.set_xticks(major_ticks)
        for i, prediction in enumerate(predictions):
            ax.plot(prediction,  label=model[i])
        ax.plot(weather_data, color='green', label='temperature', linestyle='dashed', alpha=0.2)
        avg_consumption = round(sum([sum(i) for i in zip(*predictions)])/float(len(model)), 2)
        ax.set_title('Energy Usage prediction for ' + str(arrow.now('US/Eastern').date()) + 
                     " | " + str(avg_consumption) + " kWh")
        ax.set_xlabel("Time(hrs)")
        ax.set_ylabel('kWh / C')
        ax.legend(loc="lower right")
        plt.show()

