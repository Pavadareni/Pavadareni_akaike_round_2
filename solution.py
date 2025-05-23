import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Data Loading
data_path = os.path.join('data','data.csv')
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Data Preprocessing
df.dropna(subset=['CustomerID'], inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


#Aggregate daily sales (Daily Revenue = sum of Quantity × UnitPrice per day).
df['Revenue'] = df['Quantity'] * df['UnitPrice']
daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
daily_revenue.columns = ['ds', 'y'] 


#Visualize sales trends.
plt.figure(figsize=(12, 6))
plt.plot(daily_revenue['ds'], daily_revenue['y'])
plt.title('Daily Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid()
plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/revenue_trend.png')
plt.close()

#Forecasting using Prophet
#Here we are using prohet to forecaste the next seven days of sales. The reason we are using the prohet library is that it is specifically designed for time series forecasting and can handle missing data and outliers well.
model = Prophet(daily_seasonality=True)
model.fit(daily_revenue)

# Predict total daily revenue for the next 7 days
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
fig = model.plot(forecast)

#Visualize actual vs forecasted sales.
plt.title('Forecast for Next 7 Days')
fig.savefig('outputs/forecast_plot.png')
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_csv('outputs/next_7_days_forecast.csv', index=False)
print("Forecasting complete. Results saved in 'outputs/' folder.")