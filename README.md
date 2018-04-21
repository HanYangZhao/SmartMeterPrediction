
# SmartMeterPrediction
Tool to predicted home energy usages based on machine learning and smart meter data

# SmartMeterPrediction - HydroQuebec
Tool to predicted home energy usages based on machine learning and smart meter data

# Getting Started
  * Go to the docker folder
  * Build the docker containers in the sub-folder
  * In `docker-compose.yml` add the necessary informations
  * `docker-compose up`
  * Graphana setup
    * go to `localhost:3000`
    * add the data source
    * url : http://influxdb:8086
    * Use Proxy setting
    * Database name : e
    * import `energy_usage.json` as a new dashboard
  * A new prediction should be made everyday 

# Training

  ## Pre-reqs
  * `pip install -r requirements.txt` in the training folder
  * tensorflow-gpu is strongly recommended if you have the hardware
  ## Multiple linear regression
  * Download your house's hourly datasets from Hydro-Quebec
  * Put them all in the the `./training/data/hourly` folder
  * Run the `cvs_hour_processing.py` script. This should create a new csv file.
    Make sure that there are no hours with no energy usage in the data set
  * Go to the `multi_linear_regression_hourly.py` and change `line 20` to point the new csv created
  * Run `multi_linear_regression_hourly.py`
  * Uses the temperature, hour of the day and is_workday to make predictions and train the network
  
 ## LSTM network
  * Download your house's hourly datasets from Hydro-Quebec
  * Put them all in the the `./training/data/hourly` folder
  * Run the `cvs_hour_processing.py` script
  * Make sure that there are no hours with no energy usage in the data set
  * Go to the `lstm_hourly.py` and change `line 43` to point the new csv created
  * You can adjust what parameters we want to use for training by changing `line 36` and the numbers of prediction hours in       `line 39 - line 41`
  * Uses the temperature, hour of the day previous predictions and is_workday to generate new predictions and train the network
  * NOTE: I've currently commented out in the LSTM prediction because it's not very accurate. To re-enable you need to remove the comments in `./predictor/main.py` and `./predictor/database.py`
