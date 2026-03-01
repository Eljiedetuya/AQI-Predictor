from predictor_screen import PredictorScreen
from analytics_screen import AnalyticsScreen
from ranking_screen import RankingScreen
from comparison_screen import ComparisonScreen

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Based Air Quality Forecasting System")

        self.stack = QStackedWidget()

        self.predictor = PredictorScreen(self.stack)
        self.analytics = AnalyticsScreen(self.stack)
        self.ranking = RankingScreen(self.stack)
        self.comparison = ComparisonScreen(self.stack)

        self.stack.addWidget(self.predictor)
        self.stack.addWidget(self.analytics)
        self.stack.addWidget(self.ranking)
        self.stack.addWidget(self.comparison)

        self.setCentralWidget(self.stack)