AQI Predictor PH

This is a Philippine-city AQI prediction web app built with Flask, Pandas, scikit-learn, and XGBoost. It predicts AQI from city, date, and pollutant inputs, and includes trend charts, AQI band guidance, city comparison, map snapshot, history, and export tools.

Portfolio

💡 Live  
The website is available at: https://aqi-predictor-1-g31a.onrender.com/

💻 Tech Stack:
- Python
- Flask
- Pandas / NumPy
- scikit-learn
- XGBoost
- HTML/CSS/JavaScript (Jinja templates)
- Render (deployment)

Features
- AQI prediction for supported Philippine cities
- Auto-fill pollutant defaults from historical city/month/year data
- AQI status classification (Good to Hazardous)
- 12-month trend visualization
- Pollutant contribution bars
- City comparison (up to 3 cities)
- Export result to CSV / print-to-PDF
- Local history tracking in browser

Local Development
To run the app locally, follow these steps:

1. Clone the repository to your local machine.
2. Create and activate a virtual environment.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `python app.py`
5. Open in browser:
   `http://127.0.0.1:5000/`

Deployment (Render)
1. Connect this GitHub repo to Render as a Web Service.
2. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
3. (Recommended) Environment Variable:
   - `PYTHON_VERSION=3.11.9`

Project Structure
- `app.py` - Flask app and API routes
- `templates/index.html` - Main UI
- `static/` - Static assets
- `src/` - Prediction/model utility modules
- `models/` - Trained model artifacts
- `data/` - Input dataset(s)

Notes
- This model is intended for Philippine-city use based on available training data.
- Unsupported cities are blocked in the UI/API to avoid misleading output.
