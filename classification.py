'''
Yiming Ge

This module contains all classification functions in this program.
'''

from constant import*
import yfinance as yf
import pandas as pd
import numpy as np
from regression import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score


TODAY_DATA = None
TODAY_PRICE = 0.00
TEST_DATA = None
TEST_ACTUAL_RETURN = None
TEST_ACTUAL_TARGET = None
# get data from yfinance
def get_data_classfication(ticker1, ticker2, start_date, end_date, feature1, feature2, rolling_window):
    data = yf.download(ticker1 + " " + ticker2, start=start_date, end=end_date)
    # print(data)
    # extract data and drop Null
    data_1_1 = (data[feature1][ticker1]).fillna(1)
    data_1_2 = (data[feature2][ticker1]).fillna(1)
    data_2 = (data[feature1][ticker2]).fillna(1)
    # create dataframe
    df = pd.DataFrame(columns= [ticker1, ticker2, feature2])
    df[ticker1] = data_1_1
    df[ticker2] = data_2
    df[feature2] = data_1_2
    # standard deviation
    df["STD"] = df[ticker1].rolling(rolling_window).std(ddof=0)
    # moving average
    df["MA"] = df[ticker1].rolling(rolling_window).mean()
    df = df[rolling_window:]
    df['Return'] = df[ticker1] - df[ticker1].shift(1)
    return_list = df['Return'].values.tolist()
    position_list = []
    # get position, 1 means long, 0 means short
    for ret in return_list:
        if ret >= 0:
            position_list.append(1)
        if ret < 0:
            position_list.append(0)
    position_list.append(2) # to make the position list has same length of df
    df['Target'] = position_list
    global TEST_ACTUAL_RETURN
    TEST_ACTUAL_RETURN = df["Return"][-TEST_SIZE:]
    del df["Return"]
    global TODAY_DATA 
    TODAY_DATA = normalize(df, "Target")[-1]
    global TODAY_PRICE
    TODAY_PRICE = df.iloc[-1].tolist()[0]
    print("{} TODAY PRICE: {}".format(ticker1, TODAY_PRICE))
    global TEST_DATA
    TEST_DATA = normalize(df, "Target")[-TEST_SIZE:-1]
    # print(TEST_DATA)
    global TEST_ACTUAL_TARGET
    TEST_ACTUAL_TARGET = df["Target"][-TEST_SIZE:-1]

    df = df.reset_index(drop=True)
    df = df.iloc[:-1 , :]
    # print(df)
    return df

def make_naive_bayes(x_train, x_test, y_train, y_test):
    print("$$$$Naive Bayes Classification$$$$")
    nb = GaussianNB()
    X = x_train
    Y = y_train
    nb.fit(X, Y)
  
    y_predict = nb.predict(x_test)
    f_1 = f1_score(y_test, y_predict)
    print("The f1 score for naive bayes classfication is", f_1)
    test_predict = nb.predict(TEST_DATA)
    draw_true_predict_classfication(TEST_ACTUAL_RETURN, TEST_ACTUAL_TARGET, test_predict, "Naive Bayes Classification")

    target = nb.predict([TODAY_DATA])
    
    if target == 1:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")

def make_random_forest(x_train, x_test, y_train, y_test):
    print("$$$$Random Forest Classification$$$$")
    rf = RandomForestClassifier(n_estimators=10, random_state=88,max_depth=4)
    X = x_train
    Y = y_train
    rf.fit(X, Y)
  
    y_predict = rf.predict(x_test)
    f_1 = f1_score(y_test, y_predict)
    print("The f1 score for random forest classfication is", f_1)
    test_predict = rf.predict(TEST_DATA)
    draw_true_predict_classfication(TEST_ACTUAL_RETURN, TEST_ACTUAL_TARGET, test_predict, "Random Forest Classification")

    target = rf.predict([TODAY_DATA])
    
    if target == 1:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")

def make_knn(x_train, x_test, y_train, y_test):
    print("$$$$KNN Classfication$$$$")
    knn = KNeighborsClassifier(n_neighbors=3)
    X = x_train
    Y = y_train
    knn.fit(X, Y)
  
    y_predict = knn.predict(x_test)
    f_1 = f1_score(y_test, y_predict)
    print("The f1 score for knn classfication is", f_1)
    test_predict = knn.predict(TEST_DATA)
    draw_true_predict_classfication(TEST_ACTUAL_RETURN, TEST_ACTUAL_TARGET, test_predict, "KNN Classification")

    target = knn.predict([TODAY_DATA])
    
    if target == 1:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")


def make_svm(x_train, x_test, y_train, y_test):
    print("$$$$Support Vectore Classification$$$$")
    svc =  SVC()
    X = x_train
    Y = y_train
    svc.fit(X, Y)
  
    y_predict = svc.predict(x_test)
    f_1 = f1_score(y_test, y_predict)
    print("The f1 score for SVC is", f_1)
    test_predict = svc.predict(TEST_DATA)
    draw_true_predict_classfication(TEST_ACTUAL_RETURN, TEST_ACTUAL_TARGET, test_predict, "SVC")

    target = svc.predict([TODAY_DATA])
    
    if target == 1:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")


def draw_true_predict_classfication(true_return, true_target, predict, name, p=0):
  df_pred = pd.DataFrame(true_target.values, columns=['Actual'], index=true_target.index)
  df_pred["Actual"] = df_pred["Actual"].shift(1)
  df_pred['Predicted'] = predict
  df_pred['Predicted'] = df_pred['Predicted'].shift(1)
  df_pred['Return'] = true_return

  actual_pos_list = df_pred['Actual'].values.tolist()[1:]
  predict_pos_list = df_pred['Predicted'].values.tolist()[1:]
  actual_return_list = df_pred['Return'].values.tolist()[1:]
  profit_list = []
  accurate_number = 0
  true_false_list = []
  # calculate profit
  for pos, pred_pos, r in zip(actual_pos_list, predict_pos_list, actual_return_list):
    if pos == pred_pos:
      p += abs(r)
      accurate_number += 1
      true_false_list.append(abs(r))
    if pos != pred_pos:
      p -= abs(r)
      true_false_list.append(-abs(r))
    profit_list.append(p)
  true_false_list.append(100) # make same length
  print("Last {} days, highest profit is ${}, lowest profit is ${}, final profit is ${}".format(TEST_SIZE, max(profit_list), min(profit_list),p))
  # draw diagram
  profit_list.append(2)
  df_pred["Profit"] = profit_list
  df_pred["Profit"] = df_pred["Profit"].shift(1)
#   print(df_pred)
  df_pred["MissOrTarget"] = true_false_list
  df_pred["MissOrTarget"] = df_pred["MissOrTarget"].shift(1)
  # df_pred[['Actual', 'Predicted']].plot(kind='bar')
  df_pred["MissOrTarget"].plot(kind="bar")
  plt.title("{} Miss Or Target Graph".format(name))
  plt.show()

  df_pred["Profit"].plot()
  plt.title("{} Profit and Loss Graph".format(name))
  plt.show()
  
  accurate_rate = accurate_number/len(actual_pos_list)
  print("Last {} days, long short position accurate rate is {}".format(TEST_SIZE, accurate_rate))