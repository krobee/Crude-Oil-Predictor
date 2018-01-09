# Crude Oil Predictor
A system that predicts crude oil price based on GDELT dataset
## Dataset
* GDELT events dataset
* WTI crude oil price dataset
## Details
* used Google BigQuery to filter out useful data from GDELT
* used HDFS to store datasets
* utilized Spark to read and pre-process the dataset
* applied Spark-ML to train a linear regression model  
* assessd the accuracy with R square and RMSE
