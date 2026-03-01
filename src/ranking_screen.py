from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class RiskRanking(QWidget):
    def __init__(self, data):
        super().__init__()
        layout = QVBoxLayout()

        avg = data.groupby('city_name')['AQI'].mean()
        ranked = avg.sort_values(ascending=False)

        text = "City Risk Ranking:\n\n"
        for i,(city,val) in enumerate(ranked.items(),1):
            text += f"{i}. {city} - {val:.2f}\n"

        layout.addWidget(QLabel(text))
        self.setLayout(layout)