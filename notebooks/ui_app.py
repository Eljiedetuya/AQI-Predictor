import sys
import os
import pandas as pd
# Ensure project root is on sys.path so `src` package imports work when run from notebooks/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtChart import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import Qt, QStringListModel, QTimer
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QComboBox,
    QListView, QHBoxLayout, QProgressBar, QMessageBox, QStackedWidget
)
from src.predict import predict_aqi as ml_predict_aqi

class AQICalculator(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("AQI Calculator")
        self.setGeometry(100, 100, 400, 300)
        self.parent = parent

        layout = QVBoxLayout()
        
        self.aqi_input = QLineEdit(self)
        self.aqi_input.setPlaceholderText("Enter AQI Value")
        layout.addWidget(self.aqi_input)

        self.calculate_button = QPushButton("Check AQI", self)
        self.calculate_button.clicked.connect(self.check_aqi)
        layout.addWidget(self.calculate_button)

        self.result_label = QLabel(self)
        layout.addWidget(self.result_label)

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.back_to_main)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def check_aqi(self):
        try:
            aqi_value = float(self.aqi_input.text())
            description = self.get_aqi_description(aqi_value)
            self.result_label.setText(description)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid numeric AQI value.")

    def get_aqi_description(self, aqi):
        if aqi >= 0 and aqi <= 50:
            return "AQI: {} - Good\nAir quality is considered satisfactory, and air pollution poses little or no risk.".format(aqi)
        elif aqi >= 51 and aqi <= 100:
            return "AQI: {} - Moderate\nAir quality is acceptable; however, some pollutants may pose moderate health concerns for a very small number of people.".format(aqi)
        elif aqi >= 101 and aqi <= 150:
            return "AQI: {} - Unhealthy for Sensitive Groups\nMembers of sensitive groups may experience health effects. The general public is unlikely to be affected.".format(aqi)
        elif aqi >= 151 and aqi <= 200:
            return "AQI: {} - Unhealthy\nEveryone may begin to experience health effects; members of sensitive groups may experience more serious health effects.".format(aqi)
        elif aqi >= 201 and aqi <= 300:
            return "AQI: {} - Very Unhealthy\nHealth alert: everyone may experience more serious health effects.".format(aqi)
        elif aqi >= 301:
            return "AQI: {} - Hazardous\nHealth warnings of emergency conditions. The entire population is very likely to be affected.".format(aqi)
        else:
            return "Invalid AQI value. Please enter a value greater than or equal to 0."

    def back_to_main(self):
        self.parent.show_main()  # Return to the main menu

class HistoricalAQIAnalysis(QWidget):
    def __init__(self, pollutant_data, parent):
        super().__init__()
        self.setWindowTitle("Historical AQI Analysis")
        self.setGeometry(100, 100, 900, 600)
        
        self.pollutant_data = pollutant_data
        self.parent = parent
        self.layout = QVBoxLayout()

        self.city_label = QLabel("Select City:")
        self.city_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.city_label)

        self.city_input = QComboBox(self)
        self.layout.addWidget(self.city_input)

        self.analyze_button = QPushButton("Analyze Historical AQI", self)
        self.analyze_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.analyze_button.clicked.connect(self.analyze_historical_aqi)
        self.layout.addWidget(self.analyze_button)

        self.chart_view = QChartView(self)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumHeight(300)
        self.layout.addWidget(self.chart_view)

        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #FF5733; color: white;")
        self.back_button.clicked.connect(self.back_to_main)
        self.layout.addWidget(self.back_button)

        self.setLayout(self.layout)
        self.load_city_names()

    def load_city_names(self):
        self.city_input.addItems(self.pollutant_data['city_name'].unique().tolist())

    def analyze_historical_aqi(self):
        city_name = self.city_input.currentText()
        historical_aqi_data = self.pollutant_data[self.pollutant_data['city_name'] == city_name]

        if historical_aqi_data.empty:
            QMessageBox.warning(self, "No Data", f"No historical AQI data available for {city_name}.")
            return

        self.create_historical_aqi_chart(historical_aqi_data)

    def create_historical_aqi_chart(self, historical_aqi_data):
        bar_set = QBarSet("AQI Values")
        values = historical_aqi_data['AQI'].values
        months = historical_aqi_data['Month'].astype(str).values

        bar_set.append(values)

        series = QBarSeries()
        series.append(bar_set)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Historical AQI for City")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.setBackgroundBrush(QColor("#f3f3f3"))

        axis_x = QBarCategoryAxis()
        axis_x.append(months)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, max(values) + 10)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        self.chart_view.setChart(chart)

    def back_to_main(self):
        self.parent.show_main()

class OverallAQIAnalysis(QWidget):
    def __init__(self, pollutant_data, parent):
        super().__init__()
        self.setWindowTitle("Overall AQI Analysis")
        self.setGeometry(100, 100, 900, 600)

        self.pollutant_data = pollutant_data
        self.parent = parent
        self.layout = QVBoxLayout()

        self.analyze_button = QPushButton("Analyze Overall AQI", self)
        self.analyze_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.analyze_button.clicked.connect(self.analyze_overall_aqi)
        self.layout.addWidget(self.analyze_button)

        self.chart_view = QChartView(self)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.chart_view)

        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #FF5733; color: white;")
        self.back_button.clicked.connect(self.back_to_main)
        self.layout.addWidget(self.back_button)

        self.setLayout(self.layout)

    def analyze_overall_aqi(self):
        cities = self.pollutant_data['city_name'].unique()
        actual_aqi_data = []
        predicted_aqi_data = []

        for city in cities:
            for month in range(1, 13):
                actual_aqi = self.pollutant_data[(self.pollutant_data['city_name'] == city) & (self.pollutant_data['Month'] == month)]
                if not actual_aqi.empty:
                    actual_aqi_value = actual_aqi['AQI'].mean()
                    actual_aqi_data.append(actual_aqi_value)
                else:
                    actual_aqi_data.append(0)

                predicted_aqi = self.simple_predictor(month, city)
                predicted_aqi_data.append(predicted_aqi)

        self.create_overall_aqi_chart(cities, actual_aqi_data, predicted_aqi_data)

    def simple_predictor(self, month_index, city_name):
        city_data = self.pollutant_data[self.pollutant_data['city_name'].str.strip().str.lower() == city_name.lower()]
        if city_data.empty:
            return 0  # Default value if no data is found

        historical_aqi = city_data['AQI'].mean()
        month_averages = city_data.groupby('Month')['AQI'].mean()
        month_adjusted_aqi = month_averages.get(month_index, historical_aqi)
        return month_adjusted_aqi

    def create_overall_aqi_chart(self, cities, actual_aqi_data, predicted_aqi_data):
        bar_set_actual = QBarSet("Actual AQI")
        bar_set_predicted = QBarSet("Predicted AQI")

        bar_set_actual.append(actual_aqi_data)
        bar_set_predicted.append(predicted_aqi_data)

        series = QBarSeries()
        series.append(bar_set_actual)
        series.append(bar_set_predicted)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Overall AQI Analysis for All Cities (Actual vs Predicted)")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundBrush(QColor("#f3f3f3"))

        axis_x = QBarCategoryAxis()
        axis_x.append([f"{city}" for city in cities])
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        max_value = max(max(actual_aqi_data), max(predicted_aqi_data))
        axis_y.setRange(0, max_value + 10)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        bar_set_actual.setColor(QColor(0, 0, 255))      # Blue for actual AQI
        bar_set_predicted.setColor(QColor(255, 165, 0))  # Orange for predicted AQI

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        self.chart_view.setChart(chart)

    def back_to_main(self):
        self.parent.show_main()

class AQIPredictor(QWidget):
    def __init__(self, pollutant_data, parent):
        super().__init__()
        self.setWindowTitle("AQI Predictor")
        self.setGeometry(100, 100, 900, 600)

        self.pollutant_data = pollutant_data
        self.parent = parent

        self.setStyleSheet("background-color: #f8f9fa;")

        main_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.city_input = QLineEdit(self)
        self.city_input.setPlaceholderText("Start typing city name...")
        self.city_input.setStyleSheet("padding: 10px; font-size: 16px; border: 1px solid #007BFF; border-radius: 5px;")
        input_layout.addWidget(QLabel("Enter City Name:"))
        input_layout.addWidget(self.city_input)

        self.suggestion_list = QListView(self)
        self.suggestion_list.setVisible(False)
        self.suggestion_list.setStyleSheet("padding: 10px;")
        input_layout.addWidget(self.suggestion_list)

        self.cities = self.load_city_names()
        self.city_model = QStringListModel(self.cities)
        self.suggestion_list.setModel(self.city_model)
        self.city_input.textChanged.connect(self.update_suggestions)

        main_layout.addLayout(input_layout)

        self.month_selector = QComboBox(self)
        self.month_selector.setStyleSheet("font-size: 16px; padding: 10px; border: 1px solid #007BFF; border-radius: 5px;")
        self.month_selector.addItems([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        main_layout.addWidget(QLabel("Select Month of 2025:"))
        main_layout.addWidget(self.month_selector)

        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("Predict AQI", self)
        self.predict_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #FFC107; color: black;")
        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #FF5733; color: white;")
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.back_button)
        main_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.output_label = QLabel(self)
        self.output_label.setWordWrap(True)
        main_layout.addWidget(self.output_label)

        self.chart_view = QChartView(self)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumHeight(300)
        main_layout.addWidget(self.chart_view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setLayout(main_layout)

        self.predict_button.clicked.connect(self.start_prediction)
        self.clear_button.clicked.connect(self.clear_inputs)
        self.suggestion_list.clicked.connect(self.select_suggestion)
        self.back_button.clicked.connect(self.back_to_main)

    def load_city_names(self):
        return self.pollutant_data['city_name'].unique().tolist()

    def update_suggestions(self):
        text = self.city_input.text().lower()
        filtered_cities = [city for city in self.cities if text in city.lower()]
        self.city_model.setStringList(filtered_cities)
        self.suggestion_list.setVisible(bool(filtered_cities))
        
    def select_suggestion(self, index):
        self.city_input.setText(self.city_model.data(index, Qt.DisplayRole))
        self.suggestion_list.setVisible(False)

    def start_prediction(self):
        city_name = self.city_input.text().strip()
        if not city_name or city_name.lower() not in [c.lower() for c in self.cities]:
            QMessageBox.warning(self, "Input Error", "Please enter a valid city name.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        QTimer.singleShot(100, lambda: self.progress_bar.setValue(25))
        QTimer.singleShot(200, lambda: self.progress_bar.setValue(50))
        QTimer.singleShot(300, lambda: self.progress_bar.setValue(75))
        QTimer.singleShot(400, lambda: self.progress_bar.setValue(100))
        QTimer.singleShot(500, self.predict_aqi)

    def clear_inputs(self):
        self.city_input.clear()
        self.output_label.clear()
        self.chart_view.setChart(QChart())
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def predict_aqi(self):
        city_name = self.city_input.text().strip()
        selected_month = self.month_selector.currentText()
        month_index = self.month_selector.currentIndex() + 1

        # Get actual AQI if available
        actual_rows = self.pollutant_data[self.pollutant_data['city_name'] == city_name]
        actual_aqi = actual_rows['AQI'].values[0] if not actual_rows.empty else None

        # Find pollutant row for the selected month (if available)
        city_data = self.pollutant_data[self.pollutant_data['city_name'].str.strip().str.lower() == city_name.lower()]
        month_data = city_data[city_data['Month'] == month_index]

        if not month_data.empty:
            pollutant_data = month_data.iloc[0][[
                'components.co', 'components.no', 'components.no2', 'components.o3',
                'components.so2', 'components.pm2_5', 'components.pm10', 'components.nh3'
            ]].to_dict()
        else:
            # No historical month data — use zeros for prediction but inform the user
            pollutant_data = {
                'components.co': 0,
                'components.no': 0,
                'components.no2': 0,
                'components.o3': 0,
                'components.so2': 0,
                'components.pm2_5': 0,
                'components.pm10': 0,
                'components.nh3': 0
            }
            self.output_label.setText(f"<b>No historical data for {city_name} in {selected_month} 2025; using default values.</b>")

        # Call the ML predictor (fallback to simple predictor if ML fails)
        try:
            ml_pred = ml_predict_aqi(city_name, month_index, pollutant_data)
            if ml_pred is None:
                predicted_aqi = self.simple_predictor(month_index, city_name)
            else:
                predicted_aqi = ml_pred
        except Exception:
            predicted_aqi = self.simple_predictor(month_index, city_name)

        # Prepare pollutant display
        pollutants_html = "<br>".join(
            f"<b>{k}:</b> {v}" for k, v in pollutant_data.items()
        )

        actual_aqi_display = actual_aqi if actual_aqi is not None else "N/A"

        self.output_label.setText(
            f"<b>City:</b> {city_name}<br>"
            f"<b>Actual AQI:</b> {actual_aqi_display}<br>"
            f"<b>Predicted AQI:</b> <span style='color:red;'>{predicted_aqi:.2f}</span> ({selected_month} 2025)<br><br>"
            f"<b>Pollutant Details:</b><br>{pollutants_html}"
        )

        # If actual AQI available, use it for chart; else use predicted only
        chart_actual = actual_aqi if actual_aqi is not None else predicted_aqi
        self.create_aqi_chart(chart_actual, predicted_aqi, selected_month)

    def simple_predictor(self, month_index, city_name):
        city_data = self.pollutant_data[self.pollutant_data['city_name'].str.strip().str.lower() == city_name.lower()]
        if city_data.empty:
            return 55  # Default base AQI if no data is available

        historical_aqi = city_data['AQI'].mean()
        month_averages = city_data.groupby('Month')['AQI'].mean()
        month_adjusted_aqi = month_averages.get(month_index, historical_aqi)
        return month_adjusted_aqi

    def create_aqi_chart(self, actual_aqi, predicted_aqi, selected_month):
        bar_set_actual = QBarSet("Actual AQI")
        bar_set_predicted = QBarSet("Predicted AQI")
        bar_set_actual.append([actual_aqi])
        bar_set_predicted.append([predicted_aqi])

        bar_set_actual.setColor(QColor(0, 0, 255))   # Blue for actual AQI
        bar_set_predicted.setColor(QColor(255, 0, 0))  # Red for predicted AQI

        series = QBarSeries()
        series.append(bar_set_actual)
        series.append(bar_set_predicted)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle(f"AQI Comparison for {selected_month} 2025")
        chart.setBackgroundBrush(QColor("#f3f3f3"))

        axis_x = QBarCategoryAxis()
        axis_x.append(["AQI"])
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, max(actual_aqi, predicted_aqi) + 10)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        self.chart_view.setChart(chart)

    def back_to_main(self):
        self.parent.show_main()

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Application")
        self.setGeometry(100, 100, 1000, 600)

        try:
            # Load your CSV data (try common locations: project root and data/ folder)
            filename = '20242Monthlydata_modified.csv'
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            candidates = [
                os.path.join(project_root, 'data', filename),
                os.path.join(project_root, filename),
                os.path.join(os.path.dirname(__file__), filename),
                os.path.join(os.path.dirname(__file__), '..', filename)
            ]

            found = None
            for p in candidates:
                if os.path.exists(p):
                    found = p
                    break

            if not found:
                raise FileNotFoundError(f"CSV not found. Tried paths: {candidates}")

            self.pollutant_data = pd.read_csv(found)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load the CSV file: {str(e)}")
            sys.exit(1)

        self.pollutant_data.columns = self.pollutant_data.columns.str.strip()  # Strip whitespace from columns

        self.central_widget = QStackedWidget(self)
        self.setCentralWidget(self.central_widget)

        self.main_screen = QWidget()  
        self.aqi_predictor_screen = AQIPredictor(self.pollutant_data, self)  
        self.historical_aqi_analysis_screen = HistoricalAQIAnalysis(self.pollutant_data, self)  
        self.overall_aqi_analysis_screen = OverallAQIAnalysis(self.pollutant_data, self)  
        self.aqi_calculator_screen = AQICalculator(self)  # New AQI Calculator screen

        self.central_widget.addWidget(self.main_screen)
        self.central_widget.addWidget(self.aqi_predictor_screen)
        self.central_widget.addWidget(self.historical_aqi_analysis_screen)
        self.central_widget.addWidget(self.overall_aqi_analysis_screen)
        self.central_widget.addWidget(self.aqi_calculator_screen)  # Add AQI Calculator to widget stack

        main_layout = QVBoxLayout()

        self.aqi_predictor_button = QPushButton("AQI Predictor", self)
        self.aqi_predictor_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.aqi_predictor_button.clicked.connect(self.show_aqi_predictor)

        self.historical_aqi_analysis_button = QPushButton("Historical AQI Analysis", self)
        self.historical_aqi_analysis_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.historical_aqi_analysis_button.clicked.connect(self.show_historical_aqi_analysis)

        self.overall_aqi_analysis_button = QPushButton("Overall AQI Analysis", self)
        self.overall_aqi_analysis_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.overall_aqi_analysis_button.clicked.connect(self.show_overall_aqi_analysis)

        self.aqi_calculator_button = QPushButton("AQI Calculator", self)
        self.aqi_calculator_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #007BFF; color: white;")
        self.aqi_calculator_button.clicked.connect(self.show_aqi_calculator)

        main_layout.addWidget(self.aqi_predictor_button)
        main_layout.addWidget(self.historical_aqi_analysis_button)
        main_layout.addWidget(self.overall_aqi_analysis_button)
        main_layout.addWidget(self.aqi_calculator_button)  # Include the new button

        self.main_screen.setLayout(main_layout)

        self.apply_styles()  # Apply styles

        self.show()

    def show_aqi_predictor(self):
        self.central_widget.setCurrentWidget(self.aqi_predictor_screen)

    def show_historical_aqi_analysis(self):
        self.central_widget.setCurrentWidget(self.historical_aqi_analysis_screen)

    def show_overall_aqi_analysis(self):
        self.central_widget.setCurrentWidget(self.overall_aqi_analysis_screen)

    def show_aqi_calculator(self):
        self.central_widget.setCurrentWidget(self.aqi_calculator_screen)  # Show the AQI Calculator screen

    def show_main(self):
        self.central_widget.setCurrentWidget(self.main_screen)

    def apply_styles(self):
        style = """
        QWidget {
            font-family: Helvetica, Arial, sans-serif;
            background-color: #f5f5f5;
        }
        QPushButton {
            font-size: 16px;
            background-color: #007BFF;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: none;
            transition: background-color 0.3s ease;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QLabel {
            font-size: 18px;
            margin-bottom: 10px;
            color: #333;
        }
        QLineEdit, QComboBox {
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
        QProgressBar {
            text-align: center;
            background-color: #e9ecef;
        }
        QProgressBar::chunk {
            background-color: #007BFF;
        }
        """
        self.setStyleSheet(style)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    sys.exit(app.exec_())
