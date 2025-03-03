import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.7%_ML-4.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_53.7%_UO-9.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, return_data=False):
    """
    Run the XGBoost model predictions
    
    Args:
        data: Normalized team data
        todays_games_uo: Over/under values for today's games
        frame_ml: DataFrame with game data
        games: List of games ([home_team, away_team])
        home_team_odds: Money line odds for home teams
        away_team_odds: Money line odds for away teams
        kelly_criterion: Boolean to enable Kelly Criterion calculations
        return_data: If True, return prediction data instead of printing to console
        
    Returns:
        If return_data is True, returns a dictionary with prediction results
        Otherwise, prints results to console and returns None
    """
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    # If we want to return data for UI display
    if return_data:
        predictions = []
        expected_values = []
        
        count = 0
        for game in games:
            home_team = game[0]
            away_team = game[1]
            winner = int(np.argmax(ml_predictions_array[count]))
            under_over = int(np.argmax(ou_predictions_array[count]))
            winner_confidence = ml_predictions_array[count]
            un_confidence = ou_predictions_array[count]
            
            if winner == 1:
                winner_team = home_team
                winner_confidence = round(winner_confidence[0][1] * 100, 1)
            else:
                winner_team = away_team
                winner_confidence = round(winner_confidence[0][0] * 100, 1)
                
            if under_over == 0:
                ou_pick = "UNDER"
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
            else:
                ou_pick = "OVER"
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                
            # Add prediction info
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'winner': winner_team,
                'winner_confidence': winner_confidence,
                'ou_pick': ou_pick,
                'ou_confidence': un_confidence,
                'ou_value': todays_games_uo[count] if count < len(todays_games_uo) else 0
            })
            
            # Calculate expected values
            ev_home = ev_away = 0
            if count < len(home_team_odds) and count < len(away_team_odds) and home_team_odds[count] and away_team_odds[count]:
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
            
            # Add Kelly Criterion if enabled
            home_kelly = away_kelly = 0
            if kelly_criterion:
                home_kelly = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1]) if count < len(home_team_odds) and home_team_odds[count] else 0
                away_kelly = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0]) if count < len(away_team_odds) and away_team_odds[count] else 0
            
            expected_values.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_ev': ev_home,
                'away_ev': ev_away,
                'home_kelly': home_kelly,
                'away_kelly': away_kelly
            })
            
            count += 1
            
        return {
            'predictions': predictions,
            'expected_values': expected_values
        }
    
    # Original console output functionality
    else:
        count = 0
        for game in games:
            home_team = game[0]
            away_team = game[1]
            winner = int(np.argmax(ml_predictions_array[count]))
            under_over = int(np.argmax(ou_predictions_array[count]))
            winner_confidence = ml_predictions_array[count]
            un_confidence = ou_predictions_array[count]
            if winner == 1:
                winner_confidence = round(winner_confidence[0][1] * 100, 1)
                if under_over == 0:
                    un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                    print(
                        Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                        Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                            todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
                else:
                    un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                    print(
                        Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                        Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                            todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                winner_confidence = round(winner_confidence[0][0] * 100, 1)
                if under_over == 0:
                    un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                    print(
                        Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                        Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                            todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
                else:
                    un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                    print(
                        Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                        Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                            todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            count += 1

        if kelly_criterion:
            print("------------Expected Value & Kelly Criterion-----------")
        else:
            print("---------------------Expected Value--------------------")
        count = 0
        for game in games:
            home_team = game[0]
            away_team = game[1]
            ev_home = ev_away = 0
            if home_team_odds[count] and away_team_odds[count]:
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
            expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
            bankroll_descriptor = ' Fraction of Bankroll: '
            bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
            bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

            print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
            print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
            count += 1

        deinit()