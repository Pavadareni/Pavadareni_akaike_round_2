Tasks to Complete
1. Data Preprocessing
2. Forecasting

1) Data Preprocessing
We ask the user to upload the cvs file. Load the data using pandas (read_csv).
For handling the missing values I have used dropna. Here we only check the CustomerID. If the costomer id is null,it will remove the column. Then i have convert the Invoice Date to the pandas format
Then we are using Aggregate for the daily sales report, we are grouping by the invoice date and we are adding the revenue for that particular date and reset the index. Setting the revenue colums as ds and y. ds refers the month (ex: 2011-01) and y is the output(revenue)
Then we are vizulizing the analyzed pattern 

2) Forecasting
Here we are using **Prohet** to forecaste the next seven days of sales. The reason we are using the prohet library is that it is specifically designed for time series forecasting and can handle missing data and outliers well, it is also a lightweight method.
make_future_dataframe method is used for setting the time frame of next 7 days and we are vizulizing the forcast. 

Directories
Solution
\solution.py 
This file contains the termial based code that works on the command prompt.
\sol.py
This file contains the UI for the project that is connected to the streamlit.

Output\
forecast_plot.png
It is the vizulazation of the next 7 days. Forcasting vizulization

revenue_trend.png
It is the vizulization of the patterns and trend analysis

next_7_days_forecast.csv
Contains the ds,yhat,yhat_lower,yhat_upper for next 7 days. The values of the model predicted. 


Tech Stack used
pandas for data analysis 
matplotlib for vizulization 
Prophet for model to the task forcasting 
streamlit for the UI 



 
