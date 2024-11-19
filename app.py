import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from joblib import load
# Load the model
model = load("flight_rf.joblib")


# App title
st.title("Flight Price Prediction App")

# Sidebar for user input
st.sidebar.header("Flight Details")

# Date of Journey
date_dep = st.sidebar.text_input("Departure Time (YYYY-MM-DD HH:MM)", "2024-01-01 12:00")
date_arr = st.sidebar.text_input("Arrival Time (YYYY-MM-DD HH:MM)", "2024-01-01 15:00")

# Total Stops
Total_stops = st.sidebar.selectbox("Number of Stops", [0, 1, 2, 3, 4])

# Airline Selection
airline = st.sidebar.selectbox(
    "Airline",
    [
        "Jet Airways",
        "IndiGo",
        "Air India",
        "Multiple carriers",
        "SpiceJet",
        "Vistara",
        "GoAir",
        "Multiple carriers Premium economy",
        "Jet Airways Business",
        "Vistara Premium economy",
        "Trujet",
    ],
)

# Source Selection
Source = st.sidebar.selectbox(
    "Source",
    ["Delhi", "Kolkata", "Mumbai", "Chennai"]
)

# Destination Selection
Destination = st.sidebar.selectbox(
    "Destination",
    ["Cochin", "Delhi", "New_Delhi", "Hyderabad", "Kolkata"]
)

# Prediction logic
if st.sidebar.button("Predict"):
    try:
        # Extract and transform input features
        Journey_day = int(pd.to_datetime(date_dep).day)
        Journey_month = int(pd.to_datetime(date_dep).month)
        Dep_hour = int(pd.to_datetime(date_dep).hour)
        Dep_min = int(pd.to_datetime(date_dep).minute)
        Arrival_hour = int(pd.to_datetime(date_arr).hour)
        Arrival_min = int(pd.to_datetime(date_arr).minute)
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # One-hot encoding for airlines
        airline_dict = {
            "Jet Airways": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "IndiGo": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Air India": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "Multiple carriers": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "SpiceJet": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "Vistara": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "GoAir": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "Multiple carriers Premium economy": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "Jet Airways Business": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "Vistara Premium economy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "Trujet": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        airline_features = airline_dict[airline]

        # One-hot encoding for sources
        source_dict = {
            "Delhi": [1, 0, 0, 0],
            "Kolkata": [0, 1, 0, 0],
            "Mumbai": [0, 0, 1, 0],
            "Chennai": [0, 0, 0, 1],
        }
        source_features = source_dict[Source]

        # One-hot encoding for destinations
        destination_dict = {
            "Cochin": [1, 0, 0, 0, 0],
            "Delhi": [0, 1, 0, 0, 0],
            "New_Delhi": [0, 0, 1, 0, 0],
            "Hyderabad": [0, 0, 0, 1, 0],
            "Kolkata": [0, 0, 0, 0, 1],
        }
        destination_features = destination_dict[Destination]

        # Combine all features
        features = [
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
        ] + airline_features + source_features + destination_features

        # Make prediction
        prediction = model.predict([features])
        output = round(prediction[0], 2)

        # Display the result
        st.success(f"Predicted Flight Price: Rs. {output}")

    except Exception as e:
        st.error(f"Error: {e}")
