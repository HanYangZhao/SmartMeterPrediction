# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 00:21:56 2018

@author: han
"""

from influxdb import InfluxDBClient
import pandas as pd
import arrow

class database:
        
    def __init__(self,ip, port,db_name):
        self.__client = InfluxDBClient(host=ip, port=port, database=db_name)
        
    def write_timeseries(self,values,start_date,name,measurement_name=None):
        m_name = name
        json = []
        if measurement_name:
          m_name = measurement_name
        now = arrow.get(start_date)
        for i in range(len(values)):
            time = now.shift(hours=+i).format('YYYY-MM-DD HH:mm:ss')
            data = self.datapoint_template(name,time,values[i],m_name)
            json.append(data)
        self.__client.write_points(json)
        
    def write_datapoint(self,value,start_date,name):
        now = arrow.get(start_date).format('YYYY-MM-DD HH:mm:ss')
        data = self.datapoint_template(name,now,value,name)
        print(data)
        self.__client.write_points([data])
               
    def datapoint_template(self,name,time,value,measurement_name):
        data = {
            "measurement": measurement_name,
            "time": time,
            "fields": {
                name: value
            }
        }
        return data

    def write_predicted(self,predictions,weather,start_date,models):
        json = []
        now = arrow.get(start_date)
        for i in range(len(weather)):
            time = now.shift(hours=+i).format('YYYY-MM-DD HH:mm:ss')
#            prediction = [predictions[0][i],predictions[1][i],predictions[2][i]]
            prediction = [predictions[0][i]]
            data = self.predicted_template(models,time,prediction,weather[i])
            self.__client.write_points([data], protocol='json')
        print(json)

    def predicted_template(self,models,time,predictions,temperature):
        data = {
            "measurement": "predictions",
            "time": time,
            "fields": {
                models[0]: predictions[0],
#                models[1]: predictions[1],
#                models[2]: predictions[2],
                "temperature" : temperature
            }
        }
        return data            
        
