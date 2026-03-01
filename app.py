from pathlib import Path
import re
import unicodedata

import pandas as pd
from flask import Flask, jsonify, request, render_template

from src.predict import predict_aqi

app = Flask(__name__, template_folder='templates', static_folder='static')


def normalize_city_name(value: str) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_diacritics = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9]+", " ", without_diacritics.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def load_supported_cities() -> tuple[list[str], set[str]]:
    data_candidates = [
        Path(__file__).resolve().parent / "data" / "20242Monthlydata_modified.csv",
        Path(__file__).resolve().parent / "20242Monthlydata_modified.csv",
    ]

    for path in data_candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path, usecols=["city_name"])
        cities = sorted({str(city).strip() for city in df["city_name"].dropna() if str(city).strip()})
        normalized = {normalize_city_name(city) for city in cities}
        return cities, normalized

    return [], set()


SUPPORTED_CITIES, SUPPORTED_CITY_KEYS = load_supported_cities()

POLLUTANT_COLUMN_MAP = {
    "pm25": "components.pm2_5",
    "pm10": "components.pm10",
    "no2": "components.no2",
    "so2": "components.so2",
    "co": "components.co",
    "o3": "components.o3",
    "no": "components.no",
    "nh3": "components.nh3",
}


def load_pollutant_data() -> pd.DataFrame:
    data_candidates = [
        Path(__file__).resolve().parent / "data" / "20242Monthlydata_modified.csv",
        Path(__file__).resolve().parent / "20242Monthlydata_modified.csv",
    ]

    required_cols = [
        "city_name",
        "Year",
        "Month",
        "AQI",
        "coord.lat",
        "coord.lon",
        *POLLUTANT_COLUMN_MAP.values(),
    ]
    for path in data_candidates:
        if not path.exists():
            continue
        return pd.read_csv(path, usecols=required_cols)
    return pd.DataFrame(columns=required_cols)


POLLUTANT_DATA = load_pollutant_data()
if not POLLUTANT_DATA.empty:
    POLLUTANT_DATA["city_norm"] = POLLUTANT_DATA["city_name"].astype(str).apply(normalize_city_name)


def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_city_rows(city_name: str) -> tuple[pd.DataFrame, str]:
    normalized_city = normalize_city_name(city_name)
    city_rows = POLLUTANT_DATA[POLLUTANT_DATA["city_norm"] == normalized_city] if not POLLUTANT_DATA.empty else pd.DataFrame()
    return city_rows, normalized_city


def get_default_pollutants(city_name: str, month: int, year: int) -> tuple[dict, str] | tuple[None, None]:
    city_rows, _ = get_city_rows(city_name)
    if city_rows.empty:
        return None, None

    exact_rows = city_rows[(city_rows["Month"] == month) & (city_rows["Year"] == year)]
    month_rows = city_rows[city_rows["Month"] == month]

    source = "exact_city_month_year"
    selected_rows = exact_rows
    if selected_rows.empty and not month_rows.empty:
        source = "city_month_fallback"
        selected_rows = month_rows
    if selected_rows.empty:
        source = "city_overall_fallback"
        selected_rows = city_rows

    defaults = {}
    for short_key, col_name in POLLUTANT_COLUMN_MAP.items():
        value = float(selected_rows[col_name].mean())
        defaults[short_key] = round(value, 2)

    return defaults, source


def validate_supported_city(city_name: str) -> tuple[bool, str]:
    normalized_city = normalize_city_name(city_name)
    if SUPPORTED_CITY_KEYS and normalized_city not in SUPPORTED_CITY_KEYS:
        return False, normalized_city
    return True, normalized_city


@app.route("/")
def index():
    return render_template("index.html", supported_cities=SUPPORTED_CITIES)


@app.route("/predict", methods=["POST"])
def predict_post():
    try:
        data = request.get_json(silent=True) or {}
        city_name = str(data.get("city_name", "")).strip()
        month = int(data.get("month", 1))
        year = int(data.get("year", 2025))
        pollutants_dict = data.get("pollutants_dict", {})

        if not city_name:
            return jsonify({"error": "City name is required."}), 400

        is_supported, _ = validate_supported_city(city_name)
        if not is_supported:
            return jsonify(
                {
                    "error": (
                        "This model is validated for Philippine cities only. "
                        "Please enter a city included in the Philippine training data."
                    )
                }
            ), 400

        aqi = predict_aqi(city_name=city_name, month=month, year=year, pollutants_dict=pollutants_dict)
        if aqi is None:
            return jsonify({"error": "Prediction failed"}), 400

        return jsonify(
            {
                "city": city_name,
                "month": month,
                "year": year,
                "AQI": round(aqi, 2),
                "status": classify_aqi(aqi),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pollutants/defaults", methods=["GET"])
def pollutant_defaults():
    city_name = str(request.args.get("city_name", "")).strip()
    month = request.args.get("month", type=int)
    year = request.args.get("year", type=int)

    if not city_name:
        return jsonify({"error": "City name is required."}), 400
    if month is None:
        return jsonify({"error": "Month is required."}), 400
    if year is None:
        return jsonify({"error": "Year is required."}), 400

    is_supported, _ = validate_supported_city(city_name)
    if not is_supported:
        return jsonify({"error": "Unsupported city for this Philippine-specific model."}), 400

    defaults, source = get_default_pollutants(city_name, month, year)
    if defaults is None:
        return jsonify({"error": "No pollutant data found for this city."}), 404

    return jsonify(
        {
            "city": city_name,
            "month": month,
            "year": year,
            "source": source,
            "pollutants_dict": defaults,
        }
    )


@app.route("/trend", methods=["GET"])
def city_trend():
    city_name = str(request.args.get("city_name", "")).strip()
    if not city_name:
        return jsonify({"error": "City name is required."}), 400

    is_supported, _ = validate_supported_city(city_name)
    if not is_supported:
        return jsonify({"error": "Unsupported city for this Philippine-specific model."}), 400

    city_rows, _ = get_city_rows(city_name)
    if city_rows.empty:
        return jsonify({"error": "No AQI trend data found for this city."}), 404

    grouped = city_rows.groupby("Month", as_index=False)["AQI"].mean().sort_values("Month")
    trend = [{"month": int(row["Month"]), "aqi": round(float(row["AQI"]), 2)} for _, row in grouped.iterrows()]
    return jsonify({"city": city_name, "trend": trend})


@app.route("/city/coords", methods=["GET"])
def city_coords():
    city_name = str(request.args.get("city_name", "")).strip()
    if not city_name:
        return jsonify({"error": "City name is required."}), 400

    is_supported, _ = validate_supported_city(city_name)
    if not is_supported:
        return jsonify({"error": "Unsupported city for this Philippine-specific model."}), 400

    city_rows, _ = get_city_rows(city_name)
    if city_rows.empty:
        return jsonify({"error": "No coordinate data found for this city."}), 404

    lat = float(city_rows["coord.lat"].mean())
    lon = float(city_rows["coord.lon"].mean())
    return jsonify({"city": city_name, "lat": round(lat, 6), "lon": round(lon, 6)})


@app.route("/compare", methods=["POST"])
def compare_cities():
    try:
        data = request.get_json(silent=True) or {}
        cities = data.get("cities", [])
        month = int(data.get("month", 1))
        year = int(data.get("year", 2025))

        if not isinstance(cities, list) or not cities:
            return jsonify({"error": "Please provide at least one city to compare."}), 400

        results = []
        for city in cities[:3]:
            city_name = str(city).strip()
            if not city_name:
                continue

            is_supported, _ = validate_supported_city(city_name)
            if not is_supported:
                results.append({"city": city_name, "error": "Unsupported city"})
                continue

            defaults, source = get_default_pollutants(city_name, month, year)
            if defaults is None:
                results.append({"city": city_name, "error": "No city data"})
                continue

            aqi = predict_aqi(city_name=city_name, month=month, year=year, pollutants_dict=defaults)
            if aqi is None:
                results.append({"city": city_name, "error": "Prediction failed"})
                continue

            results.append(
                {
                    "city": city_name,
                    "aqi": round(float(aqi), 2),
                    "status": classify_aqi(float(aqi)),
                    "source": source,
                }
            )

        return jsonify({"month": month, "year": year, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["GET"])
def predict_get():
    aqi = predict_aqi(
        city_name="Manila",
        month=1,
        year=2025,
        pollutants_dict={
            "pm25": 12,
            "pm10": 25,
            "no2": 18,
            "so2": 5,
            "co": 0.4,
            "o3": 30,
        },
    )
    return jsonify({"AQI": round(aqi, 2), "Status": "Good"})


if __name__ == "__main__":
    app.run(debug=True)
