# AQI Predictor PH

[![Live on Render](https://img.shields.io/badge/Live-Render-46E3B7?logo=render&logoColor=white)](https://aqi-predictor-1-g31a.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Modeling](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-F7931E)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-HTML%20%7C%20CSS%20%7C%20JS-1572B6)](#)

Philippine-city AQI prediction web app built with Flask, Pandas, scikit-learn, and XGBoost.  
It predicts AQI from city, date, and pollutant inputs and provides trend, comparison, map snapshot, history, and export features.

## Live
App URL: [https://aqi-predictor-1-g31a.onrender.com/](https://aqi-predictor-1-g31a.onrender.com/)

## Tech Stack
- Python
- Flask
- Pandas / NumPy
- scikit-learn
- XGBoost
- HTML / CSS / JavaScript (Jinja templates)
- Render (deployment)

## Features
- AQI prediction for supported Philippine cities
- Auto-fill pollutant defaults from historical city/month/year data
- AQI status classification (Good to Hazardous)
- 12-month trend visualization
- Pollutant contribution bars
- City comparison (up to 3 cities)
- Export result to CSV / print-to-PDF
- Local prediction history in browser

## Local Development
1. Clone the repository:
   ```bash
   git clone https://github.com/Eljiedetuya/AQI-Predictor.git
   cd AQI-Predictor
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open:
   `http://127.0.0.1:5000/`

## Deployment (Render)
Set your Render Web Service with:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Environment Variable (recommended): `PYTHON_VERSION=3.11.9`

## Project Structure
- `app.py` - Flask app and API routes
- `templates/index.html` - Main UI
- `static/` - Static assets
- `src/` - Prediction/model utility modules
- `models/` - Trained model artifacts
- `data/` - Input dataset(s)

## Notes
- This model is intended for Philippine-city use based on available training data.
- Unsupported cities are blocked in the UI/API to reduce misleading outputs.
