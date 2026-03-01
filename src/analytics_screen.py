from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtChart import QBarSeries, QBarSet, QChart, QChartView

class FeatureImportanceScreen(QWidget):
    def __init__(self, model):
        super().__init__()
        layout = QVBoxLayout()

        importances = model.rf_model.feature_importances_
        bar = QBarSet("Importance")
        bar.append(importances.tolist())

        series = QBarSeries()
        series.append(bar)

        chart = QChart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setTitle("Feature Importance")

        view = QChartView(chart)
        layout.addWidget(view)

        self.setLayout(layout)