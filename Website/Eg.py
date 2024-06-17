from flask import Flask, render_template, request, redirect, url_for
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from keras.optimizers import Adam


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Perform prediction or any other processing here
    # Redirect to the appropriate location
    start = '2015-01-01'
    end = '2023-12-31'

    st.title('Stock Trend Prediction')
    user_input = st.text_input('Enter Stock Ticker','AAPL')
    #df = data.DataReader(user_input, 'yahoo', start, end)
    df = yf.download(user_input, start=start, end=end)

    #Description of Data
    st.subheader('Data from 2015 - 2023')
    st.write(df.describe())

    #Visualization
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart With 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart With 100MA and 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    #Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)
    #Splitting Data into x_train and y_train
    #x_train = []
    #y_train = []

    #for i in range(100, data_training_array.shape[0]):
    #   x_train.append(data_training_array[i-100: i])
    #  y_train.append(data_training_array[i,0])

    #x_train, y_train = np.array(x_train), np.array(y_train)
    #Load My Model

    model = load_model('keras_model.keras')
    #optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate as needed
    #model.compile(optimizer='adam', loss='mean_squared_error')

    #Testing Part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        #Final Graph
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        #plt.show()
        st.pyplot(fig2)


        return redirect(url_for('result'))

        @app.route('/result')

        def result():
         # Display prediction result or any other content
         return render_template('result.html')

         if __name__ == '__main__':
            app.run(debug=True)
