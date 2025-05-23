import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

st.set_page_config(page_title="Sales Forecasting App", layout="centered")
st.title(" Sales Forecasting")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    daily_revenue.columns = ['ds', 'y']


    st.subheader(" Daily Revenue Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_revenue['ds'], daily_revenue['y'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.set_title("Daily Revenue Over Time")
    st.pyplot(fig)

    with st.spinner("Training forecasting model..."):
        model = Prophet(daily_seasonality=True)
        model.fit(daily_revenue)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

    st.subheader("Forecast for Next 7 Days")
    fig_forecast = model.plot(forecast)
    st.pyplot(fig_forecast)

    st.subheader(" Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).reset_index(drop=True))

    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Forecast CSV",
        data=csv,
        file_name='next_7_days_forecast.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a data.csv file to proceed.")