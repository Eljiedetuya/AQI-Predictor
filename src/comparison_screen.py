from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class ModelComparison(QWidget):
    def __init__(self, model):
        super().__init__()
        layout = QVBoxLayout()

        text = (
            f"Hybrid Model Performance\n\n"
            f"MAE: {model.mae:.2f}\n"
            f"R2 Score: {model.r2:.2f}"
        )

        layout.addWidget(QLabel(text))
        self.setLayout(layout)