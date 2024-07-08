from flask import Flask, jsonify
from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

# MongoDB connection
mongo_uri = "mongodb+srv://fsiddiqui107:gc79mKY4g6hGrbVL@ssnscluster.fsy0znp.mongodb.net/?retryWrites=true&w=majority&appName=SSNSCluster"
client = MongoClient(mongo_uri)

# Select the database and collection
db = client['testing']
collection = db['testing']

def preprocess_data():
    # Load data from MongoDB
    data = list(collection.find({}, {'_id': 0}))

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Convert timestamp to datetime in UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)

    return df

def fit_arima_model(df):
    # Fit ARIMA model (simple example, you may need to tune order)
    model = ARIMA(df['temperature'], order=(5, 1, 0))  # Change order as needed
    model_fit = model.fit()

    return model_fit

def predict_arima(model_fit, future_steps):
    # Predict the next value
    prediction = model_fit.forecast(steps=future_steps)
    return prediction

def get_predictions(df, last_timestamp):
    model_fit = fit_arima_model(df)
    
    intervals = {
        "15 mins": 15,
        "30 mins": 30,
        "1 hour": 60
    }
    
    predictions = {}
    for key, minutes in intervals.items():
        future_time = last_timestamp + timedelta(minutes=minutes)
        prediction = predict_arima(model_fit, minutes).iloc[-1]
        future_time_str = future_time.strftime('%Y-%m-%d')
        time_str = future_time.strftime('%H:%M:%S')

        predictions[key] = {
            "date": future_time_str,
            "humidity": f"{prediction:.2f} %",
            "perticulate_matter": f"{prediction:.2f} µg/m³",
            "pressure": f"{prediction:.2f} hPa",
            "temperature": f"{prediction:.2f} °C",
            "time": time_str
        }
    
    return predictions

@app.route('/data', methods=['GET'])
def get_data():
    df = preprocess_data()
    data = df.reset_index().to_dict(orient='records')
    return jsonify(data)

@app.route('/predictions', methods=['GET'])
def get_predictions_route():
    df = preprocess_data()
    last_timestamp = df.index[-1]  # Get the last timestamp from the DataFrame
    predictions = get_predictions(df, last_timestamp)
    
    # Check if there is data within the last minute
    current_time_utc = datetime.now(timezone.utc)
    last_minute_data = df[df.index > (current_time_utc - timedelta(minutes=1))]
    if not last_minute_data.empty:
        sensor_working = "Yes"
    else:
        sensor_working = "No"
    
    predictions["Sensor Working?"] = sensor_working
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
