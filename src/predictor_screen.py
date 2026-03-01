from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtChart import QChart, QLineSeries, QChartView
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
from report_generator import generate_pdf

class AQIPredictor(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.layout = QVBoxLayout()
        self.label = QLabel("AQI Predictor")
        self.layout.addWidget(self.label)

        self.predict_btn = QPushButton("Predict & Show Trend")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.layout.addWidget(self.predict_btn)

        self.chart_view = QChartView()
        self.layout.addWidget(self.chart_view)

        self.setLayout(self.layout)

    def run_prediction(self):
        city_data = self.model.data.iloc[-1:]
        prediction = self.model.predict(city_data[self.model.features])[0]

        self.label.setText(
            f"Predicted AQI: {prediction:.2f}\n"
            f"MAE: {self.model.mae:.2f}\n"
            f"R2: {self.model.r2:.2f}"
        )

        self.show_trend_chart()

        generate_pdf(prediction, self.model.mae, self.model.r2)

    def show_trend_chart(self):
        series = QLineSeries()

        last_row = self.model.data.iloc[-1]
        for month in range(1, 13):
            input_data = last_row[self.model.features].copy()
            input_data['Month'] = month
            input_data['month_sin'] = np.sin(2*np.pi*month/12)
            input_data['month_cos'] = np.cos(2*np.pi*month/12)

            df = pd.DataFrame([input_data])
            pred = self.model.predict(df)[0]
            series.append(month, pred)

        chart = QChart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setTitle("12-Month Forecast Trend")

        self.chart_view.setChart(chart)