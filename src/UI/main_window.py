from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QTableWidget, QTableWidgetItem, QPushButton, 
                            QComboBox, QTabWidget, QGridLayout, QFrame, QCheckBox,
                            QScrollArea, QSplitter, QMessageBox, QProgressBar,
                            QStatusBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon

from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.UI.charts import GamePredictionWidget

class PredictionThread(QThread):
    """Thread for running predictions in background"""
    prediction_finished = pyqtSignal(dict)
    progress_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_type, data, uo, frame, games, home_odds, away_odds, use_kelly=False):
        super().__init__()
        self.model_type = model_type
        self.data = data
        self.uo = uo
        self.frame = frame
        self.games = games
        self.home_odds = home_odds
        self.away_odds = away_odds
        self.use_kelly = use_kelly
        
    def run(self):
        try:
            self.progress_update.emit(10)
            # This is used to track our results
            results = {
                'games': self.games,
                'predictions': [],
                'expected_values': []
            }
            
            self.progress_update.emit(30)
            
            # Run models and get output
            if self.model_type == "xgb":
                results = self._run_xgboost()
            elif self.model_type == "nn":
                results = self._run_nn()
                
            self.progress_update.emit(100)
            self.prediction_finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Error running predictions: {str(e)}")
    
    def _run_xgboost(self):
        """Run the XGBoost model and return results"""
        return XGBoost_Runner.xgb_runner(
            self.data, 
            self.uo, 
            self.frame, 
            self.games, 
            self.home_odds, 
            self.away_odds, 
            self.use_kelly,
            return_data=True
        )
    
    def _run_nn(self):
        """Run the Neural Network model and return results"""
        return NN_Runner.nn_runner(
            self.data, 
            self.uo, 
            self.frame, 
            self.games, 
            self.home_odds, 
            self.away_odds, 
            self.use_kelly,
            return_data=True
        )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NBA ML Sports Betting")
        self.resize(1200, 800)
        
        # Setup UI
        self._setup_ui()
        
        # Initialize data structures
        self.odds = {}
        self.games = []
        self.todays_games_uo = []
        self.home_team_odds = []
        self.away_team_odds = []
        self.data = None
        self.frame_ml = None
        
        # Load initial data
        self._load_data()
    
    def _setup_ui(self):
        # Main widget
        main_widget = QWidget()
        main_widget.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a2e; color: #ffffff; }
            QLabel { color: #ffffff; }
            QPushButton { 
                background-color: #4361ee; 
                color: white; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3a56d4; }
            QPushButton:pressed { background-color: #2c3e8c; }
            QTableWidget { 
                background-color: #252546; 
                color: #ffffff;
                gridline-color: #3a3a5c;
                border: 1px solid #3a3a5c;
                border-radius: 4px;
            }
            QTableWidget::item { padding: 4px; }
            QTableWidget::item:selected { background-color: #4361ee; }
            QHeaderView::section { 
                background-color: #333355; 
                color: white;
                padding: 4px;
                border: 1px solid #3a3a5c;
            }
            QComboBox {
                background-color: #252546;
                color: white;
                padding: 6px;
                border: 1px solid #3a3a5c;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #252546;
                color: white;
                selection-background-color: #4361ee;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a5c;
                border-radius: 4px;
                background-color: #252546;
            }
            QTabBar::tab {
                background-color: #333355;
                color: white;
                padding: 8px 16px;
                border: 1px solid #3a3a5c;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4361ee;
            }
            QCheckBox {
                color: white;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QProgressBar {
                border: 1px solid #3a3a5c;
                border-radius: 4px;
                text-align: center;
                background-color: #252546;
            }
            QProgressBar::chunk {
                background-color: #4361ee;
                width: 10px;
            }
            QSplitter::handle {
                background-color: #3a3a5c;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #252546;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3a5c;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header section
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("🏀 NBA ML Sports Betting")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #4361ee;")
        header_layout.addWidget(title_label, 1)
        
        # Controls section - right side of header
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        
        # Sportsbook selector
        sportsbook_label = QLabel("Sportsbook:")
        self.sportsbook_combo = QComboBox()
        self.sportsbook_combo.addItems(["fanduel", "draftkings", "betmgm", "pointsbet", "caesars"])
        self.sportsbook_combo.setMinimumWidth(120)
        
        # Load button
        load_button = QPushButton("Load Odds")
        load_button.setIcon(QIcon.fromTheme("view-refresh"))
        load_button.clicked.connect(self._load_data)
        
        controls_layout.addWidget(sportsbook_label)
        controls_layout.addWidget(self.sportsbook_combo)
        controls_layout.addWidget(load_button)
        
        header_layout.addWidget(controls_widget)
        main_layout.addWidget(header_widget)
        
        # Model controls
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model selection buttons
        self.xgb_button = QPushButton("Run XGBoost Model")
        self.xgb_button.setMinimumWidth(180)
        self.nn_button = QPushButton("Run Neural Network Model")
        self.nn_button.setMinimumWidth(180)
        self.kelly_checkbox = QCheckBox("Show Kelly Criterion")
        
        self.xgb_button.clicked.connect(lambda: self._run_predictions("xgb"))
        self.nn_button.clicked.connect(lambda: self._run_predictions("nn"))
        
        model_layout.addWidget(self.xgb_button)
        model_layout.addWidget(self.nn_button)
        model_layout.addWidget(self.kelly_checkbox)
        model_layout.addStretch()
        
        main_layout.addWidget(model_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # Games table
        games_widget = QWidget()
        games_layout = QVBoxLayout(games_widget)
        games_layout.setContentsMargins(0, 0, 0, 0)
        
        games_header = QLabel("Today's Games")
        games_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        games_layout.addWidget(games_header)
        
        self.games_table = QTableWidget()
        self.games_table.setColumnCount(7)
        self.games_table.setHorizontalHeaderLabels([
            "Away Team", "Home Team", "Spread", "Moneyline Away", 
            "Moneyline Home", "Over/Under", "Date"
        ])
        self.games_table.horizontalHeader().setStretchLastSection(True)
        self.games_table.setAlternatingRowColors(True)
        games_layout.addWidget(self.games_table)
        
        main_splitter.addWidget(games_widget)
        
        # Predictions area
        self.predictions_tabs = QTabWidget()
        
        # XGBoost tab
        self.xgb_tab = QWidget()
        xgb_layout = QVBoxLayout(self.xgb_tab)
        xgb_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create scroll area for prediction widgets
        xgb_scroll = QScrollArea()
        xgb_scroll.setWidgetResizable(True)
        
        # Container for prediction widgets
        self.xgb_predictions_container = QWidget()
        self.xgb_predictions_layout = QVBoxLayout(self.xgb_predictions_container)
        self.xgb_predictions_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.xgb_predictions_layout.setSpacing(20)
        
        xgb_scroll.setWidget(self.xgb_predictions_container)
        xgb_layout.addWidget(xgb_scroll)
        
        # Neural Network tab
        self.nn_tab = QWidget()
        nn_layout = QVBoxLayout(self.nn_tab)
        nn_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create scroll area for prediction widgets
        nn_scroll = QScrollArea()
        nn_scroll.setWidgetResizable(True)
        
        # Container for prediction widgets  
        self.nn_predictions_container = QWidget()
        self.nn_predictions_layout = QVBoxLayout(self.nn_predictions_container)
        self.nn_predictions_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.nn_predictions_layout.setSpacing(20)
        
        nn_scroll.setWidget(self.nn_predictions_container)
        nn_layout.addWidget(nn_scroll)
        
        # Add tabs
        self.predictions_tabs.addTab(self.xgb_tab, "XGBoost Predictions")
        self.predictions_tabs.addTab(self.nn_tab, "Neural Network Predictions")
        
        main_splitter.addWidget(self.predictions_tabs)
        main_splitter.setSizes([300, 500])  # Set initial sizes
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def _load_data(self):
        """Load odds data for the selected sportsbook"""
        sportsbook = self.sportsbook_combo.currentText()
        self.status_bar.showMessage(f"Loading odds from {sportsbook}...")
        
        try:
            # Fetch odds data
            odds_provider = SbrOddsProvider(sportsbook=sportsbook)
            self.odds = odds_provider.get_odds()
            
            if not self.odds:
                self.status_bar.showMessage(f"No odds data available from {sportsbook}")
                return
                
            self.games = create_todays_games_from_odds(self.odds)
            
            if not self.games:
                self.status_bar.showMessage(f"No games found in odds data from {sportsbook}")
                return
            
            # Get today's games over/under and money line odds
            self.todays_games_uo = []
            self.home_team_odds = []
            self.away_team_odds = []
            
            for game in self.games:
                home_team = game[0]
                away_team = game[1]
                game_key = f"{home_team}:{away_team}"
                
                if game_key in self.odds:
                    game_odds = self.odds[game_key]
                    
                    # Get over/under
                    if 'under_over_odds' in game_odds:
                        self.todays_games_uo.append(game_odds['under_over_odds'])
                    else:
                        self.todays_games_uo.append(0)
                    
                    # Get money line odds
                    if home_team in game_odds and 'money_line_odds' in game_odds[home_team]:
                        self.home_team_odds.append(game_odds[home_team]['money_line_odds'])
                    else:
                        self.home_team_odds.append(0)
                        
                    if away_team in game_odds and 'money_line_odds' in game_odds[away_team]:
                        self.away_team_odds.append(game_odds[away_team]['money_line_odds'])
                    else:
                        self.away_team_odds.append(0)
            
            # Update games table
            self._update_games_table()
            
            # Get team data for predictions
            self._fetch_team_data()
            
            self.status_bar.showMessage(f"Loaded {len(self.games)} games from {sportsbook}")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading odds: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to load odds data: {str(e)}")
    
    def _fetch_team_data(self):
        """Fetch team statistics for predictions"""
        try:
            # API URL for team stats
            data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
                    'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
                    'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
                    'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
                    'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
                    'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
                    'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
                    'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            
            # Get team stats
            data = get_json_data(data_url)
            self.team_df = to_data_frame(data)
            
            # Calculate days rest for teams
            self._calculate_days_rest()
            
            # Create data needed for predictions
            self._prepare_prediction_data()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error fetching team data: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to fetch team statistics: {str(e)}")
    
    def _calculate_days_rest(self):
        """Calculate days rest for teams in today's games"""
        self.home_team_days_rest = []
        self.away_team_days_rest = []
        
        try:
            # In a real implementation, this would fetch the NBA schedule
            # and calculate days rest like in main.py
            # For this example, we'll use placeholder values
            for game in self.games:
                self.home_team_days_rest.append(2)  # Placeholder value
                self.away_team_days_rest.append(1)  # Placeholder value
        except Exception as e:
            self.status_bar.showMessage(f"Error calculating days rest: {str(e)}")
            # Use default values
            self.home_team_days_rest = [2] * len(self.games)
            self.away_team_days_rest = [1] * len(self.games)
    
    def _prepare_prediction_data(self):
        """Prepare data for prediction models"""
        try:
            # Create match data (similar to createTodaysGames in main.py)
            match_data = []
            
            for i, game in enumerate(self.games):
                home_team = game[0]
                away_team = game[1]
                
                if home_team not in team_index_current or away_team not in team_index_current:
                    continue
                
                home_days_rest = self.home_team_days_rest[i] if i < len(self.home_team_days_rest) else 2
                away_days_rest = self.away_team_days_rest[i] if i < len(self.away_team_days_rest) else 1
                
                home_team_series = self.team_df.iloc[team_index_current.get(home_team)]
                away_team_series = self.team_df.iloc[team_index_current.get(away_team)]
                
                stats = pd.concat([home_team_series, away_team_series])
                stats['Days-Rest-Home'] = home_days_rest
                stats['Days-Rest-Away'] = away_days_rest
                match_data.append(stats)
            
            games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
            games_data_frame = games_data_frame.T
            
            self.frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
            data = self.frame_ml.values
            data = data.astype(float)
            
            # For XGBoost, use data as is
            self.data = data
            # For NN, normalize data
            self.data_normalized = tf.keras.utils.normalize(data, axis=1)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error preparing prediction data: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to prepare prediction data: {str(e)}")
    
    def _update_games_table(self):
        """Update the games table with current odds data"""
        self.games_table.setRowCount(0)  # Clear table
        
        # Add rows for each game
        for i, game in enumerate(self.games):
            self.games_table.insertRow(i)
            
            home_team = game[0]
            away_team = game[1]
            
            # Get odds info
            game_key = f"{home_team}:{away_team}"
            
            # Set cell values
            self.games_table.setItem(i, 0, QTableWidgetItem(away_team))
            self.games_table.setItem(i, 1, QTableWidgetItem(home_team))
            
            if game_key in self.odds:
                game_odds = self.odds[game_key]
                
                # Add spread if available
                if 'spread' in game_odds:
                    self.games_table.setItem(i, 2, QTableWidgetItem(str(game_odds['spread'])))
                
                # Add money line odds
                if away_team in game_odds and 'money_line_odds' in game_odds[away_team]:
                    away_ml = game_odds[away_team]['money_line_odds']
                    ml_item = QTableWidgetItem(str(away_ml))
                    if away_ml > 0:
                        ml_item.setForeground(QColor("#4ade80"))  # Green for positive
                    else:
                        ml_item.setForeground(QColor("#f87171"))  # Red for negative
                    self.games_table.setItem(i, 3, ml_item)
                
                if home_team in game_odds and 'money_line_odds' in game_odds[home_team]:
                    home_ml = game_odds[home_team]['money_line_odds']
                    ml_item = QTableWidgetItem(str(home_ml))
                    if home_ml > 0:
                        ml_item.setForeground(QColor("#4ade80"))  # Green for positive
                    else:
                        ml_item.setForeground(QColor("#f87171"))  # Red for negative
                    self.games_table.setItem(i, 4, ml_item)
                
                # Add over/under
                if 'under_over_odds' in game_odds:
                    uo = game_odds['under_over_odds']
                    self.games_table.setItem(i, 5, QTableWidgetItem(str(uo)))
            
            # Today's date
            today = datetime.today().strftime("%Y-%m-%d")
            self.games_table.setItem(i, 6, QTableWidgetItem(today))
        
        # Resize columns to content
        self.games_table.resizeColumnsToContents()
    
    def _run_predictions(self, model_type):
        """Run predictions with the selected model"""
        if not hasattr(self, 'games') or not self.games:
            self.status_bar.showMessage("No games loaded. Please load odds first.")
            QMessageBox.warning(self, "Warning", "No games loaded. Please load odds data first.")
            return
            
        if not hasattr(self, 'data') or self.data is None:
            self.status_bar.showMessage("Team data not loaded. Please try reloading odds.")
            QMessageBox.warning(self, "Warning", "Team data not loaded. Please try reloading odds.")
            return
        
        # Update UI
        self.status_bar.showMessage(f"Running {model_type.upper()} predictions...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Disable buttons while predicting
        self.xgb_button.setEnabled(False)
        self.nn_button.setEnabled(False)
        
        # Create prediction thread
        input_data = self.data_normalized if model_type == "nn" else self.data
        
        self.pred_thread = PredictionThread(
            model_type, 
            input_data, 
            self.todays_games_uo, 
            self.frame_ml, 
            self.games, 
            self.home_team_odds, 
            self.away_team_odds,
            self.kelly_checkbox.isChecked()
        )
        
        # Connect signals
        self.pred_thread.prediction_finished.connect(self._update_predictions)
        self.pred_thread.progress_update.connect(self.progress_bar.setValue)
        self.pred_thread.error_occurred.connect(self._handle_prediction_error)
        self.pred_thread.finished.connect(self._prediction_thread_finished)
        
        # Start thread
        self.pred_thread.start()
    
    def _handle_prediction_error(self, error_message):
        """Handle errors from prediction thread"""
        self.status_bar.showMessage(f"Error in predictions: {error_message}")
        QMessageBox.critical(self, "Prediction Error", error_message)
        
        # Re-enable buttons
        self.xgb_button.setEnabled(True)
        self.nn_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def _prediction_thread_finished(self):
        """Clean up after prediction thread finishes"""
        # Re-enable buttons
        self.xgb_button.setEnabled(True)
        self.nn_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def _update_predictions(self, results):
        """Update the predictions with graphical widgets"""
        model_type = self.sender().model_type
        
        # Determine which container to use
        if model_type == "xgb":
            container = self.xgb_predictions_container
            layout = self.xgb_predictions_layout
        else:
            container = self.nn_predictions_container
            layout = self.nn_predictions_layout
        
        # Clear existing predictions
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Add predictions for each game
        for i, pred in enumerate(results['predictions']):
            ev = results['expected_values'][i]
            
            # Create prediction widget
            game_widget = GamePredictionWidget()
            game_widget.set_data(
                pred, 
                ev, 
                self.kelly_checkbox.isChecked()
            )
            
            # Add to layout
            layout.addWidget(game_widget)
        
        # Add stretch to push widgets to top
        layout.addStretch()
        
        # Select tab with new predictions
        if model_type == "xgb":
            self.predictions_tabs.setCurrentIndex(0)
        else:
            self.predictions_tabs.setCurrentIndex(1)
            
        self.status_bar.showMessage(f"{model_type.upper()} predictions complete")