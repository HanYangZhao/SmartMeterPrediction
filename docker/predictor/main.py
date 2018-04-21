# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 00:27:18 2018

@author: han
"""

from future_predictor import Predictor
from database import database
import arrow
import requests
import numpy as np
import json
import os
import time as s

hydro_ip = str(os.environ['HYDRO_IP'])
hydro_port = str(os.environ['HYDRO_PORT'])

db_ip = str(os.environ['INFLUX_IP'])
db_port = str(os.environ['INFLUX_PORT'])
db_database = str(os.environ['INFLUXDB_DB'])

input_steps = 48
hours = int(float(os.environ['PREDICT_HOURS']))
api = str(os.environ['DARKSKY_KEY'])
lat = float(os.environ['LAT'])
lon = float(os.environ['LON'])

update_delay = str(os.environ['UPDATE_DELAY'])

model_1 = "lstm50_48hrs.h5"
model_2 = "lstm75_48hrs.h5"
model_3 = "dnn_100.h5"
# models = ["lstm50_48hrs", "lstm75_48hrs", "dnn_100"]
models = ["dnn_100"]

predict = Predictor()


def wait_for_hydro():
    req = None
    while True:
        print('not connected')
        s.sleep(5)
        try:
            req = requests.get('http://' + hydro_ip + ":" + hydro_port + '/status')
        except BaseException as exp:
            pass
        if req:
            return


if __name__ == '__main__':
    is_updated = False
    prev_date = arrow.now().date().day
    db = database(db_ip, db_port, db_database)
    wait_for_hydro()
    while True:
        s.sleep(120)
        date = arrow.now().date()
        current_date = date.day
        if is_updated and str(arrow.now().time().hour) == update_delay and int(current_date - prev_date) >= 1:
            is_updated = False
        print(is_updated)

        if is_updated == False:
            try:
                requests.get('http://' + hydro_ip + ":" + hydro_port + '/refreshData')
                past_usage = np.array(json.loads(requests.get('http://' + hydro_ip + ":" + hydro_port + '/getTwoDaysUsage').text))
                current_data = json.loads(requests.get('http://' + hydro_ip + ":" + hydro_port + '/getData').text)
            except BaseException as e:
                continue

            try:
                weather_data = predict.future_weather(api, lat, lon, hours, time=arrow.now())
            except BaseException as e:
                continue
            
            # "Hours prediction"
            #prediction_1 = predict.predict_future_lstm(hours, model_1, past_usage, weather_data, input_steps)
            #prediction_2 = predict.predict_future_lstm(hours, model_2, past_usage, weather_data, input_steps)
            prediction_3 = predict.predict_future_dnn(hours, model_3, weather_data)
            
            #predictions = [prediction_1, prediction_2, prediction_3]
            predictions = [prediction_3]

            
            now = arrow.now()
            start_date = now.replace(hour=4, minute=0, second=1, microsecond=0)
#            today_consumption = max(sum(prediction_1[0:24]), sum(prediction_2[0:24]), sum(prediction_3[0:24]))
#            next_day_consumption = max(sum(prediction_1[24:48]), sum(prediction_2[24:48]), sum(prediction_3[24:48]))
            today_consumption =  sum(prediction_3[0:24])
            next_day_consumption = sum(prediction_3[24:48])
            db.write_datapoint(today_consumption, start_date.replace(hour=4, minute=0, second=0, microsecond=0),
                               'predicted_today_consumption')
            db.write_datapoint(next_day_consumption, start_date.replace(hour=4, minute=0, second=0, microsecond=0),
                               'predicted_next_day_consumption')
            db.write_predicted(predictions, weather_data, start_date, models)
            # 7 Days prediction
            if hours >= (24 * 7):
                seven_days_consumption = sum(prediction_3[0:168])
                db.write_datapoint(seven_days_consumption, start_date.replace(hour=4, minute=0, second=0, microsecond=0),
                               'predicted_7days_consumption')     


            yesterday_usage = past_usage[24:48,0].tolist()
            yesterday = arrow.now().shift(days=-1)
            start_date = yesterday.replace(hour=4, minute=0, second=0,microsecond=0)
            db.write_timeseries(yesterday_usage, start_date, 'yesterday_hourly_usage', 'past_data')

            yesterday_temp = past_usage[24:48, 1].tolist()
            yesterday = arrow.now().shift(days=-1)
            start_date = yesterday.replace(hour=4, minute=0, second=0, microsecond=0)
            db.write_timeseries(yesterday_temp, start_date, 'yesterday_temp', 'past_data')

            start_date =  arrow.now()
            avg_daily = current_data['period_mean_daily_consumption']
            db.write_datapoint(avg_daily, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'avg_daily_consumption')

            total_consumption = current_data['period_total_consumption']
            db.write_datapoint(total_consumption, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'total_consumption')

            bill = current_data['period_total_bill']
            db.write_datapoint(bill, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'total_bill')

            daily_bill = current_data['period_mean_daily_bill']
            db.write_datapoint(daily_bill, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'daily_bill')

            current_day = current_data['period_length']
            db.write_datapoint(current_day, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'cycle')

            projection_bill = current_data['period_projection']
            db.write_datapoint(projection_bill, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'projection_bill')

            yesterday_consumption = current_data['yesterday_total_consumption']
            db.write_datapoint(yesterday_consumption, start_date.replace(hour=4, minute=0, second=0, microsecond=0), 'yesterday_consumption')
            is_updated = True
            print("updated")
            prev_date = arrow.now().date().day
