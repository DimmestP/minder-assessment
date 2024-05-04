# UK DRI Care Research and Technology Centre Data Engineering Task

Python code for the technical assessment for the role of Data Engineer on the Minder project. The task is to predict whether a house has multiple occupants or just one according to motion sensor data set across rooms of the house. 

## Getting started

The jupyter notebook with outputs is available in the python folder. The requirements.txt file can recreate the venv with all of the python packages used to analyse the data.

## Outline of analysis process

- [x] Import sqlite data in pandas
- [x] Initial data exploration (Num of houses, rooms, time points)
- [x] Data cleaning (any duplicates? missing data)
- [x] Convert event-based record to hourly counts (maybe check if half-hourly or daily is better?)
- [x] Fit multivariate time series data to different data
- [x] Test model using cross-validation
- [x] Train classifier on time series parameters
