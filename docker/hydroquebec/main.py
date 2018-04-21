from hydro_qc import HydroQuebecClient
import asyncio
import numpy as np
import arrow
from flask import Flask,abort
import json
import os

username = str(os.environ['USERNAME'])
password = str(os.environ['PASSWORD'])
contract = str(os.environ['CONTRACT'])


loop = asyncio.get_event_loop()
def hydro_setup(username,password):

    hydroQC = HydroQuebecClient(username, password, timeout=None)

    try:
        fut = asyncio.wait([hydroQC.fetch_data()])
        loop.run_until_complete(fut)
    except BaseException as exp:
        print(exp)
    return hydroQC

__hydroQC = hydro_setup(username,password)
app = Flask(__name__)


@app.route('/status')
def status():
    return "ready"


@app.route('/refreshData',methods=['GET'])
def refreshData():
    try:
        fut = asyncio.wait([__hydroQC.fetch_data()])
        loop.run_until_complete(fut)
    except BaseException as exp:
        abort(404)
    return "success"

@app.route('/getData',methods=['GET'])
def getData():
    return json.dumps(__hydroQC.get_data()[contract])
@app.route('/getTwoDaysUsage',methods=['GET'])
def getTwoDaysUsage():
    yesterday_hourly_data = next (iter (__hydroQC.get_data().values()))['yesterday_hourly_consumption']
    two_days_ago_hourly_data = next (iter (__hydroQC.get_data().values()))['two_days_ago_hourly_consumption']

    is_workday = 1
    weekday = arrow.now().weekday() - 1
    if(weekday > 4):
        is_workday = 0

    yesterday_data = np.empty([24,4])
    for i,data in enumerate(yesterday_hourly_data):
        yesterday_data[i,0] = data['total']
        yesterday_data[i, 1] = data['temp']
        yesterday_data[i, 2] = i
        yesterday_data[i, 3] = is_workday

    is_workday = 1
    weekday = arrow.now().weekday() - 2
    if(weekday > 4):
        is_workday = 0

    two_days_data = np.empty([24,4])
    for i,data in enumerate(two_days_ago_hourly_data):
        two_days_data[i,0] = data['total']
        two_days_data[i, 1] = data['temp']
        two_days_data[i, 2] = i
        two_days_data[i, 3] = is_workday
    return json.dumps((np.append(two_days_data,yesterday_data,axis=0).tolist()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
