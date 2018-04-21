docker run -it \
-e HYDRO_IP=hydro \
-e HYDRO_PORT=5000 \
-e INFLUX_IP=influxdb \
-e INFLUX_PORT=8086 \
-e INFLUX_DB=e \
-e PREDICT_HOURS=168 \
-e LON= 
-e LAT=  \
-e DARKSKY_KEY= \
--name predictor \
--entrypoint /bin/bash \
predictor_docker 

