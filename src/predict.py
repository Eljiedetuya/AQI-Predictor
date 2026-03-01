"""
Prediction script to load trained models and make predictions on new data
Usage: python predict.py
"""
import pandas as pd
import numpy as np
import os
from .model_trainer import HybridAQIModel

def predict_aqi(city_name, month, year, pollutants_dict):
    """
    Make AQI prediction for a specific city, month, and year
    
    Args:
        city_name (str): City name
        month (int): Month (1-12)
        year (int): Year (e.g., 2024, 2025)
        pollutants_dict (dict): Dictionary with pollutant values
            Keys: 'components.co', 'components.no', 'components.no2',
                  'components.o3', 'components.so2', 'components.pm2_5',
                  'components.pm10', 'components.nh3'
    
    Returns:
        float: Predicted AQI value
    """
    try:
        # Load the trained model
        print("Loading pre-trained model...")
        model = HybridAQIModel.load_model()

        # Normalize incoming pollutant keys so callers can pass short names
        key_map = {
            'pm25': 'components.pm2_5',
            'pm2_5': 'components.pm2_5',
            'pm10': 'components.pm10',
            'no2': 'components.no2',
            'no': 'components.no',
            'so2': 'components.so2',
            'co': 'components.co',
            'o3': 'components.o3',
            'nh3': 'components.nh3'
        }

        normalized = {}
        for k, v in pollutants_dict.items():
            target = key_map.get(k, k)
            normalized[target] = v

        # Create input dataframe with normalized keys
        input_data = pd.DataFrame([normalized])
        input_data['Month'] = month
        input_data['Year'] = year

        # Add periodic features for month
        input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        input_data['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Add year-based features (normalize year to 0-1 range)
        year_normalized = (year - 2020) / 10.0
        input_data['year_normalized'] = year_normalized
        input_data['year_sin'] = np.sin(2 * np.pi * year_normalized)
        input_data['year_cos'] = np.cos(2 * np.pi * year_normalized)

        # Add interactions (guard missing columns)
        pm25 = input_data.get('components.pm2_5', pd.Series([0])).iloc[0]
        pm10 = input_data.get('components.pm10', pd.Series([0])).iloc[0]
        no = input_data.get('components.no', pd.Series([0])).iloc[0]
        no2 = input_data.get('components.no2', pd.Series([0])).iloc[0]
        o3 = input_data.get('components.o3', pd.Series([0])).iloc[0]

        input_data['pm_interaction'] = pm25 * pm10
        input_data['nox_interaction'] = no * no2
        input_data['ozone_no2_ratio'] = (o3 + 1e-6) / (no2 + 1e-6)

        # Ensure all features expected by the model are present; fill missing with 0
        for feat in model.features:
            if feat not in input_data.columns:
                input_data[feat] = 0

        # Reorder to model features
        input_data = input_data[model.features]

        # Make prediction
        prediction = model.predict(input_data)[0]
        return float(prediction)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == '__main__':
    print("=" * 70)
    print("AQI PREDICTION TOOL")
    print("=" * 70)
    
    # Example: Predict AQI for Delhi in March 2025
    example_pollutants = {
        'components.co': 1.5,
        'components.no': 45.2,
        'components.no2': 38.5,
        'components.o3': 52.1,
        'components.so2': 12.3,
        'components.pm2_5': 89.5,
        'components.pm10': 156.2,
        'components.nh3': 25.8
    }
    
    print("\nExample Prediction:")
    print(f"City: Delhi")
    print(f"Month: March (3)")
    print(f"Pollutants:")
    for pollutant, value in example_pollutants.items():
        print(f"  {pollutant}: {value}")
    
    prediction = predict_aqi('Delhi', 3, 2024, example_pollutants)
    if prediction is not None:
        print(f"\n[OK] Predicted AQI: {prediction:.2f}")
    else:
        print("\n[ERROR] Prediction failed")
    
    print("\n" + "=" * 70)
    print("To use this tool:")
    print("1. Import this script in your PyQt5 app")
    print("2. Call predict_aqi(city, month, pollutants_dict)")
    print("3. Returns: float (predicted AQI value)")
    print("=" * 70)
    
