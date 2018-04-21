import pandas as pd
import arrow
import csv
import glob
files = glob.glob('*_hour_*.csv')

dfs = []
print(files)
# load the csv and user row 0 as headers
for file in files:
    df = pd.read_csv(file,sep=';',  names=["Contract", "Date and time", "kWh", "Consumption code","Average temperature (°C)","Temperature code","Month","Weekday","Hour","Workday"])

    df.drop(df.index[0], inplace=True)

    # reverse the data order because hydro quebec put the oldest data in the last row
    df = df.iloc[::-1]

    #remove the contract, consumption code, temperatrue code column
    df.drop(['Contract','Consumption code','Temperature code'], 1, inplace=True)

    df.rename(columns={'Average temperature (°C)': 'temperature'}, inplace=True)


    for i, row in df.iterrows():
        date = arrow.get(row["Date and time"], 'YYYY-MM-DD HH:mm:ss')
        df.at[i, 'Month'] = int(date.date().month / 1)
        df.at[i, 'Hour'] = int(date.datetime.hour / 1)
        df.at[i, 'Weekday'] =  int(date.weekday() / 1)
        workday = 1
        if(date.weekday() > 4):
            workday = 0
        df.at[i, 'Workday'] = workday

    dfs.append(df)
outDf = pd.concat(dfs)
name = str(arrow.get(outDf.iloc[0]['Date and time']).date()) + "_" + str(arrow.get(outDf.iloc[-1]['Date and time']).date()) + ".csv"
outDf.to_csv(name,index=False)








