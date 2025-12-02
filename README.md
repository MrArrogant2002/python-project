# NYC Taxi Fare Predictor - Streamlit App

A machine learning-powered web application to predict NYC taxi fares based on pickup/dropoff locations, time, and passenger count.

## Features

- 🚕 Real-time taxi fare predictions
- 📍 Interactive map visualization
- 🎯 User-friendly input interface
- 📊 Model performance metrics
- 🗺️ Popular NYC location presets

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Pre-trained model file: `taxi_fare_pipeline.pkl`

## Installation

1. **Activate your virtual environment** (if not already activated):
```bash
source venv/Scripts/activate  # On Windows with Git Bash
```

2. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

## Running the App

To start the Streamlit application, run:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter Pickup Details**:
   - Select pickup date and time
   - Enter pickup latitude and longitude

2. **Enter Dropoff Details**:
   - Select number of passengers (1-6)
   - Enter dropoff latitude and longitude

3. **View Results**:
   - Click the "Predict Fare" button
   - See the predicted fare amount
   - View trip summary and distance

## Input Ranges

- **Latitude**: 40.0 to 42.0 (NYC area)
- **Longitude**: -75.0 to -72.0 (NYC area)
- **Passengers**: 1 to 6
- **Date**: 2009-01-01 to 2025-12-31

## Popular NYC Locations

For convenience, the sidebar includes coordinates for popular locations:
- Times Square: (40.7580, -73.9855)
- JFK Airport: (40.6413, -73.7781)
- LaGuardia Airport: (40.7769, -73.8740)
- Central Park: (40.7829, -73.9654)
- Brooklyn Bridge: (40.7061, -73.9969)
- Statue of Liberty: (40.6892, -74.0445)

## Model Information

The app uses a pre-trained XGBoost model with the following features:
- Datetime features (year, month, day, weekday, hour)
- Haversine distance between pickup and dropoff
- Distance from major NYC landmarks
- Passenger count
- GPS coordinates

## Troubleshooting

**Model file not found error:**
- Ensure `taxi_fare_pipeline.pkl` exists in the project directory
- Run the Jupyter notebook to train and save the model if needed

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify your virtual environment is activated

**Invalid input errors:**
- Check that coordinates are within valid NYC ranges
- Ensure date/time values are reasonable

## Project Structure

```
z-python-project/
├── app.py                          # Streamlit application
├── taxi_fare_pipeline.pkl          # Pre-trained model
├── nyc-taxi-fare-pipeline.ipynb   # Training notebook
├── requirements.txt                # Python dependencies
├── train.csv                       # Training data
├── test.csv                        # Test data
└── README.md                       # This file
```

## License

This project is for educational purposes.
