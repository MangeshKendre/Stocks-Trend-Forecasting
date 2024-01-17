

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense,Dropout
import yfinance as yf
from datetime import datetime
import streamlit as st






# Define the time range
start = datetime(1996, 1, 1)
end = datetime(2024, 1, 16)


# Set page title and favicon
st.set_page_config(page_title='📈 Stocks Trend Forecasting 📉', page_icon='📈')

st.title(' 📈  Stocks Trend Forecasting 📉')
user_input=st.text_input('Enter The Stock Ticker','SBIN.NS')




# Fetch historical stock data for AAPL from Yahoo Finance using yfinance
df = yf.download(user_input, start=start, end=end)

#describing data
st.subheader('Data from 1996 to 15 th January 2024')





st.write(df.describe())


# to disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#visualization
st.subheader("Open Price vs Time chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.Open)
st.pyplot(fig)


st.subheader("Open Price vs 100 days MA vs 200 days MA")
ma100=df.Open.rolling(100).mean()
ma200=df.Open.rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.Open,'r',label='Actual Open Price ')
plt.plot(ma100,'g',label='100 days MA')
plt.plot(ma200,'y',label='200 days MA')
ax.legend()
st.pyplot(fig)


#code for testing of lstm model build
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = df[['Open']]
# Calculate the index for the split
split_index = int(0.7 * len(X))
# Split the data into training and testing sets sequentially
train_data = X.iloc[:split_index, :]
test_data = X.iloc[split_index:, :]
scaler=MinMaxScaler(feature_range=(0,1))
train_data_array=scaler.fit_transform(train_data)
x_train=[]
y_train=[]
for i in range(100,len(train_data_array)):
    x_train.append(train_data_array[i-100:i])
    y_train.append(train_data_array[i])
x_train,y_train=np.array(x_train),np.array(y_train)    
y_train = y_train.reshape(-1, 1)



model=load_model('keras_model.h5')



past100days=train_data.tail(100)
final_df=past100days.append(train_data,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test = []
y_test = []
# Change the loop condition to ensure it starts from index 100
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred=model.predict(x_test)
scaler=scaler.scale_  
scale_factor=1/scaler[0]
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor


# Plotting the actual values (y_test) and predicted values (y_pred)
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(y_test, 'r', label='Actual')
ax.plot(y_pred, 'g', label='Forecasted')
ax.set_xlabel('Time')
ax.set_ylabel('Stock Price')
ax.legend()

# Display the plot in Streamlit with headings
st.header('Actual vs Forecasted Stock Prices on Test data')
st.pyplot(fig)