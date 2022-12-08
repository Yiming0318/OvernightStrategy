'''
Yiming Ge
Stock_regression_analyzer
This module contains all the constants in this program.
You can change any constant for different result and time window
'''
# Stock You want to analyze, any leagal TICKER eg "MSFT"
from sklearn.model_selection import train_test_split


ANALYZE_TICKER = "MSFT" # MANU
START_DATE = "2012-01-01"
END_DATE = "2022-10-07"
# correlated stock Performance, here we use s&p500, you can change to other ticker eg "AAPL"
CORRELATED_TICKER = "^GSPC"
# YAHOO FINANCE give you Open, High, Low, Close, Adj Close, Volume
PRICE = "Adj Close" # PICK ADJ CLOSE AS PRICE HERE, you can change to CLOSE or OPEN
FEATURE = "Volume"  # PICK VOLUME AS FEATRUE HERE, you can change to others
# Std range period, 1 week here, you can change to any days to calculate the std
ROLLING_WINDOW = 5
# train test split parameter
SEED = 66 # change seed get different split data
TEST_PERCENT = 0.3  # test sample percentage
TEST_SIZE = 120