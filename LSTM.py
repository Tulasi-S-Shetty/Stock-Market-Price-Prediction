#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      suman
#
# Created:     17-03-2024
# Copyright:   (c) suman 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.optimizers import Adam

start = '2015-01-01'
end = '2023-12-31'

df = yf.download('TSLA', start=start, end=end)
print(df.head())
print(df.tail())

df = df.reset_index()
df.head()
#print(df.reset_index())


#To Drop a Column
df = df.drop(['Adj Close'], axis = 1)
print(df.head())

plt.plot(df.Close)
plt.show()

#If we want 100 days moving average
ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.show()

#If we want 200 days moving average
ma200 = df.Close.rolling(200).mean()
ma200


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200,  'g')
#plt.show()


#Listing How many rows and columns are there
#print(df.shape)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

#print(data_training.shape)
#print(data_testing.shape)

#It Will print the 70% of training data
#print(data_training.head())

#It will print the 30% of testing dataset
#print(data_testing.head())


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
#print(data_training_array)

#print(data_training_array.shape)


x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)


#Model
from keras.layers import Input, Dense, Dropout , LSTM
from keras.models import Sequential

#import tensorflow as tf
#model = tf.keras.Sequential()

model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
#model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))


#model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
#model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

#model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
#model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))



#model.add(LSTM(units = 120, activation = 'relu'))
#model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.summary()
model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

model.save('keras_model.keras')

#Testing Data
#print(data_testing.head())

past_100_days = data_training.tail(100)
#final_df = past_100_days.append(data_testing, ignore_index=True)
#or
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

#print(final_df.head())

input_data = scaler.fit_transform(final_df)
#print(input_data)

#print(input_data.shape)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
#print(x_test.shape)
#print(y_test.shape)

#Making Predicions

y_predicted = model.predict(x_test)
#print(y_predicted.shape)

#print(y_test)
#print(y_predicted)

#print(scaler.scale_)

scale_factor = 1/0.01186803
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

























#dataFrame = pd.read_csv("C:\\Users\\suman\\Desktop\\HTML\\Stock Market\\AAPL.csv")
#print("Stock Prediction...\n   \n",dataFrame)