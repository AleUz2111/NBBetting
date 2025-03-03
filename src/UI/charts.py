from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QLinearGradient
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, QPointF

class PredictionChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.data = {}
        self.setStyleSheet("background-color: #2D3748;")

    def set_data(self, data):
        """
        Set chart data
        
        Args:
            data: Dictionary with keys 'home_confidence', 'away_confidence', 
                 'home_team', 'away_team', etc.
        """
        self.data = data
        self.update()

    def paintEvent(self, event):
        if not self.data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up metrics
        width = self.width()
        height = self.height()
        center_x = width // 2
        chart_height = height - 60
        
        # Draw title
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        title = f"{self.data.get('away_team', '')} @ {self.data.get('home_team', '')}"
        painter.drawText(QRect(0, 10, width, 30), Qt.AlignmentFlag.AlignCenter, title)
        
        # Draw confidence bars
        self._draw_confidence_bar(painter, 
                                  center_x - 120, 
                                  60, 
                                  100, 
                                  chart_height, 
                                  self.data.get('away_confidence', 0), 
                                  self.data.get('away_team', 'Away'), 
                                  QColor(59, 130, 246))
                                  
        self._draw_confidence_bar(painter, 
                                  center_x + 20, 
                                  60, 
                                  100, 
                                  chart_height, 
                                  self.data.get('home_confidence', 0), 
                                  self.data.get('home_team', 'Home'), 
                                  QColor(239, 68, 68))
                                  
        # Draw winner indicator
        winner = self.data.get('winner', '')
        if winner:
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor("#34D399"))
            
            text = f"Prediction: {winner} wins"
            painter.drawText(QRect(0, height - 40, width, 30), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_confidence_bar(self, painter, x, y, width, height, value, label, color):
        # Draw label
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.drawText(QRect(x, y - 25, width, 20), Qt.AlignmentFlag.AlignCenter, label)
        
        # Draw background bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.drawRoundedRect(x, y, width, height, 5, 5)
        
        # Draw value bar
        bar_height = int(height * (value / 100))
        y_pos = y + height - bar_height
        
        gradient = QLinearGradient(QPointF(x, y_pos), QPointF(x, y + height))
        gradient.setColorAt(0, color)
        gradient.setColorAt(1, color.darker(150))
        
        painter.setBrush(gradient)
        painter.drawRoundedRect(x, y_pos, width, bar_height, 5, 5)
        
        # Draw value text
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRect(x, y_pos - 25, width, 20), Qt.AlignmentFlag.AlignCenter, f"{value}%")

class EVGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.ev_value = 0
        self.setStyleSheet("background-color: #2D3748;")
        self.team_name = ""

    def set_data(self, ev_value, team_name):
        self.ev_value = ev_value
        self.team_name = team_name
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up metrics
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 20
        
        # Draw team name
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRect(0, 10, width, 30), Qt.AlignmentFlag.AlignCenter, self.team_name)
        
        # Draw gauge background
        painter.setPen(QPen(QColor(255, 255, 255, 30), 10))
        painter.drawArc(center_x - radius, center_y - radius, radius * 2, radius * 2, 210 * 16, 120 * 16)
        
        # Determine color based on EV value
        if self.ev_value > 10:
            color = QColor(52, 211, 153)  # Green
        elif self.ev_value > 0:
            color = QColor(251, 191, 36)  # Yellow
        else:
            color = QColor(239, 68, 68)   # Red
            
        # Map EV value to angle (-15 to +15 maps to 210° to 330°)
        ev_range = 30  # Total range of EV to display (-15 to +15)
        clamped_ev = max(min(self.ev_value, ev_range/2), -ev_range/2)  # Clamp between -15 and +15
        angle_range = 120  # Total angle range in degrees
        
        # Convert EV to angle (mapping -15..+15 to 210°..330°)
        angle = 210 + ((clamped_ev + ev_range/2) / ev_range) * angle_range
        sweep = angle - 210
        
        # Draw value arc
        painter.setPen(QPen(color, 10))
        painter.drawArc(center_x - radius, center_y - radius, radius * 2, radius * 2, 210 * 16, int(sweep * 16))
        
        # Draw EV value text
        font = QFont("Arial", 18, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(color)
        text = f"{self.ev_value:+.1f}"  # Format with sign and 1 decimal place
        
        # Draw the value in the center
        painter.drawText(QRect(0, center_y - 15, width, 40), Qt.AlignmentFlag.AlignCenter, text)
        
        # Draw "Expected Value" label
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(QRect(0, center_y + 20, width, 20), Qt.AlignmentFlag.AlignCenter, "Expected Value")

class KellyGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 100)
        self.kelly_value = 0
        self.team_name = ""
        self.setStyleSheet("background-color: #2D3748;")

    def set_data(self, kelly_value, team_name):
        self.kelly_value = kelly_value
        self.team_name = team_name
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up metrics
        width = self.width()
        height = self.height()
        bar_height = 30
        bar_y = height // 2 - bar_height // 2
        
        # Draw team name
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRect(0, 10, width, 20), Qt.AlignmentFlag.AlignCenter, self.team_name)
        
        # Draw background bar
        bar_width = width - 40
        bar_x = (width - bar_width) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 5, 5)
        
        # Determine color based on Kelly value
        if self.kelly_value > 10:
            color = QColor(52, 211, 153)  # Green
        elif self.kelly_value > 0:
            color = QColor(251, 191, 36)  # Yellow
        else:
            color = QColor(239, 68, 68)   # Red
        
        # Draw value bar (max 100% of bar width)
        kelly_width = min(self.kelly_value / 30, 1.0) * bar_width
        painter.setBrush(color)
        if kelly_width > 0:
            painter.drawRoundedRect(bar_x, bar_y, int(kelly_width), bar_height, 5, 5)
        
        # Draw value text
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRect(0, bar_y + bar_height + 5, width, 20), 
                        Qt.AlignmentFlag.AlignCenter, 
                        f"Kelly: {self.kelly_value:.1f}%")

class GamePredictionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 400)
        self.setStyleSheet("background-color: #1F2937; border-radius: 10px;")
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Create prediction chart
        self.prediction_chart = PredictionChart()
        main_layout.addWidget(self.prediction_chart)
        
        # Create EV gauges
        ev_layout = QHBoxLayout()
        self.home_ev_gauge = EVGauge()
        self.away_ev_gauge = EVGauge()
        ev_layout.addWidget(self.away_ev_gauge)
        ev_layout.addWidget(self.home_ev_gauge)
        main_layout.addLayout(ev_layout)
        
        # Create Kelly gauges
        kelly_layout = QHBoxLayout()
        self.home_kelly_gauge = KellyGauge()
        self.away_kelly_gauge = KellyGauge()
        kelly_layout.addWidget(self.away_kelly_gauge)
        kelly_layout.addWidget(self.home_kelly_gauge)
        main_layout.addLayout(kelly_layout)
        
        # Create OU prediction label
        self.ou_label = QLabel()
        self.ou_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ou_label.setStyleSheet("color: white; font-size: 14px; padding: 10px;")
        main_layout.addWidget(self.ou_label)
        
    def set_data(self, prediction, expected_value, show_kelly=False):
        """Update widget with prediction data"""
        # Set prediction chart data
        self.prediction_chart.set_data({
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'home_confidence': prediction['winner_confidence'] if prediction['winner'] == prediction['home_team'] else 100 - prediction['winner_confidence'],
            'away_confidence': prediction['winner_confidence'] if prediction['winner'] == prediction['away_team'] else 100 - prediction['winner_confidence'],
            'winner': prediction['winner']
        })
        
        # Set EV gauge data
        self.home_ev_gauge.set_data(expected_value['home_ev'], prediction['home_team'])
        self.away_ev_gauge.set_data(expected_value['away_ev'], prediction['away_team'])
        
        # Set Kelly gauge data and visibility
        self.home_kelly_gauge.set_data(expected_value['home_kelly'], prediction['home_team'])
        self.away_kelly_gauge.set_data(expected_value['away_kelly'], prediction['away_team'])
        self.home_kelly_gauge.setVisible(show_kelly)
        self.away_kelly_gauge.setVisible(show_kelly)
        
        # Set O/U prediction text
        ou_text = f"{prediction['ou_pick']} {prediction['ou_value']} ({prediction['ou_confidence']}% confidence)"
        self.ou_label.setText(ou_text)