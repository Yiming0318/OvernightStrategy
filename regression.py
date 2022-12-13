'''
Yiming Ge

This module contains all regression functions in this program.
'''
from sklearn import linear_model
from constant import*
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

TODAY_DATA = None
TODAY_PRICE = 0.00
TEST_DATA = None
TEST_ACTUAL_PRICE = None
# get data from yfinance
def get_data(ticker1, ticker2, start_date, end_date, feature1, feature2, rolling_window):
    data = yf.download(ticker1 + " " + ticker2, start=start_date, end=end_date)
    # print(data)
    # extract data and drop Null
    data_1_1 = (data[feature1][ticker1]).fillna(1)
    data_1_2 = (data[feature2][ticker1]).fillna(1)
    data_2 = (data[feature1][ticker2]).fillna(1)
    # create dataframe
    df = pd.DataFrame(columns= [ticker1, ticker2, feature2])
    df[ticker1] = data_1_1
    # Target is tmr price
    df["Target"] = df[ticker1].shift(-1)
    df[ticker2] = data_2
    df[feature2] = data_1_2
    # standard deviation
    df["STD"] = df[ticker1].rolling(rolling_window).std(ddof=0)
    # moving average
    df["MA"] = df[ticker1].rolling(rolling_window).mean()
    df = df[rolling_window:]
    # print(df)
    global TODAY_DATA 
    TODAY_DATA = normalize(df, "Target")[-1]
    global TODAY_PRICE
    TODAY_PRICE = df.iloc[-1].tolist()[0]
    print("{} TODAY PRICE: {}".format(ticker1, TODAY_PRICE))
    global TEST_DATA
    TEST_DATA = normalize(df, "Target")[-TEST_SIZE:-1]
    # print(TEST_DATA)
    global TEST_ACTUAL_PRICE
    TEST_ACTUAL_PRICE = df["Target"][-TEST_SIZE:-1]
    # print(TEST_ACTUAL_PRICE)
    df = df.reset_index(drop=True)
    df = df.iloc[:-1 , :]
    return df

def print_data_info(ticker1, ticker2, price, start_date, end_date, feature, window):
    print("Depend value is {}'s {} ".format(ticker1, price))
    print("Time range is from [{} to {}), exclude the last date".format(start_date, end_date))
    print("Features are {}, {}, STD, MovingAverage".format(ticker2, feature))
    print("STD rolling window is {} days".format(window))

# computes the principal components of the given data
# returns the means, standard deviations, eigenvalues, eigenvectors, and projected data
def pca(data, y, normalize=True):
  # assign to A the data as a numpy matrix 
    A = data.loc[:, data.columns != y].to_numpy()
  # assign to m the mean values of the columns of A
    m = np.mean(A, axis=0)
  # assign to D the difference matrix A - m
    D = A - m
  # if normalize is true
  #    Compute the standard deviations of each column
  # else
  #    Assign all 1s to the standard deviation vector (1 for each column)
    if normalize == True:
        std = np.std(A, axis=0)
    else:
        std = 1
  # Divide each column by its standard deviation vector
  #    (hint: this can be done as a single operation)
    D = D / std 
  # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    U, S, V = np.linalg.svd(D, full_matrices=False)
  # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
  #   divided by the degrees of freedom (N-1). The values are sorted.
    eig_val = S**2 / (len(D) - 1)
  # project the data onto the eigenvectors. Treat V as a transformation 
  #   matrix and right-multiply it by D transpose. The eigenvectors of A 
  #   are the rows of V. The eigenvectors match the order of the eigenvalues.
    eig_vect = V
    project_data = D.dot(V.transpose())
    return m, std, D, eig_val, eig_vect, project_data

# print the result for all the pca info
def print_pca_result(mean, std, D, eig_val, eig_vect, project_data):
    print("mean: ", mean)
    print("std: ", std)
    print("D: \n", D)
    print("eigenvalues: \n", eig_val)
    print("eigenvectors: \n", eig_vect)
    print("projected data: \n", project_data)

# normalize x matrix
def normalize(data, y):
    A = data.loc[:, data.columns != y].to_numpy()
    m = np.mean(A, axis=0)
    D = A - m
    std = np.std(A, axis=0)
    D = D / std 
    return D

# train test split
def my_train_test_split(df, test_size_percent ,seed):
    # split the data into train and test set
    # random_state works like seed which make sure results are reproducible.
    train, test = train_test_split(df, test_size = test_size_percent, random_state=seed, shuffle=True)
    # check the training data and testing data
    # print("The first ten items of training data: \n", train.head(10))
    print("The total number of training data is: ", len(train))
    print("The percentage of training data is: {:.1%}" .format(len(train)/len(df)))
    # print("The first ten items of testing data: \n", test.head(10))
    print("The total number of testing data is: ", len(test))
    print("The percentage of testing data is: {:.1%}".format(len(test)/len(df)))
    return train, test



# multi linear regression
def make_multi_linear_regression(x_train, x_test, y_train, y_test):
    print("$$$$MULTI LINEAR REGRESSION$$$$")
    linear_model = LinearRegression()
    X = x_train
    Y = y_train
    linear_model.fit(X, Y)
    coef = [round(item, 4) for item in linear_model.coef_]
    intercept = round(linear_model.intercept_, 4)
    print("The coef for multi linear regression are ", coef)
    print("The intercept for multi linear regression is ", intercept)
    y_predict = linear_model.predict(x_test)
    r_2 = r2_score(y_test, y_predict)
    print("The r^2 for multi linear regression is", r_2)
    test_predict = linear_model.predict(TEST_DATA)
    draw_true_predict(TEST_ACTUAL_PRICE, test_predict, 0, "Multi Linear Regression")

    tmr_price = linear_model.predict([TODAY_DATA])
    print("tomorrow price is {}".format(tmr_price))
    if tmr_price - TODAY_PRICE >= 0:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")

# make ridge model
def make_ridge_regression(x_train, x_test, y_train, y_test):
    print("$$$$RIDGE REGRESSION$$$$")
    ridge_model = Ridge(alpha=1.0)
    X = x_train
    Y = y_train
    ridge_model.fit(X, Y)
    coef = [round(item, 4) for item in ridge_model.coef_]
    intercept = round(ridge_model.intercept_, 4)
    print("The coef for ridge regression are ", coef)
    print("The intercept for ridge regression is ", intercept)
    y_predict = ridge_model.predict(x_test)
    r_2 = r2_score(y_test, y_predict)
    print("The r^2 for ridge regression is", r_2)
    test_predict = ridge_model.predict(TEST_DATA)
    draw_true_predict(TEST_ACTUAL_PRICE, test_predict, 0, "Ridge Regression")
    tmr_price = ridge_model.predict([TODAY_DATA])
    print("tomorrow price is {}".format(tmr_price))
    if tmr_price - TODAY_PRICE >= 0:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")

# make lasso model
def make_lasso_regression(x_train, x_test, y_train, y_test):
    print("$$$$LASSO REGRESSION$$$$")
    lasso_model = linear_model.Lasso(alpha=0.1)
    X = x_train
    Y = y_train
    lasso_model.fit(X, Y)
    coef = [round(item, 4) for item in lasso_model.coef_]
    intercept = round(lasso_model.intercept_, 4)
    print("The coef for lasso regression are ", coef)
    print("The intercept for lasso regression is ", intercept)
    y_predict = lasso_model.predict(x_test)
    r_2 = r2_score(y_test, y_predict)
    print("The r^2 for lasso regression is", r_2)
    test_predict = lasso_model.predict(TEST_DATA)
    draw_true_predict(TEST_ACTUAL_PRICE, test_predict, 0, "Lasso Regression")
    tmr_price = lasso_model.predict([TODAY_DATA])
    print("tomorrow price is {}".format(tmr_price))
    if tmr_price - TODAY_PRICE >= 0:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")

# make random forest regressor
def make_random_forest_regressor(x_train, x_test, y_train, y_test):
    print("$$$$RANDOM FOREST REGRESSOR$$$$")
    rf_regr = RandomForestRegressor(max_depth=4, random_state=88)
    rf_regr.fit(x_train, y_train)
    y_predict = rf_regr.predict(x_test)
    r_2 = r2_score(y_test, y_predict)
    print("The r^2 for random forest regressor is", r_2)
    tmr_price = rf_regr.predict([TODAY_DATA])
    test_predict = rf_regr.predict(TEST_DATA)
    draw_true_predict(TEST_ACTUAL_PRICE, test_predict, 0, "Random Forest Regressor")
    print("tomorrow price is {}".format(tmr_price))
    if tmr_price - TODAY_PRICE >= 0:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")
    # tree.plot_tree(rf_regr.estimators_[0])
    # plt.show()


def make_knn_regressor(x_train, x_test, y_train, y_test):
    print("$$$$K Nearest Neighbors Regressor$$$$")
    knn_regr = KNeighborsRegressor(n_neighbors=3)
    knn_regr.fit(x_train, y_train)
    y_predict = knn_regr.predict(x_test)
    r_2 = r2_score(y_test, y_predict)
    print("The r^2 for KNN regressor is", r_2)
    tmr_price = knn_regr.predict([TODAY_DATA])
    test_predict = knn_regr.predict(TEST_DATA)
    draw_true_predict(TEST_ACTUAL_PRICE, test_predict, 0, "KNN Regressor")
    print("tomorrow price is {}".format(tmr_price))
    if tmr_price - TODAY_PRICE >= 0:
      print("Recommendation:Long")
    else:
      print("Recommendation:Short")


def draw_true_predict(true, predict, p, name): 
  # p is the starting profit
  # name is the regression name
  df_pred = pd.DataFrame(true.values, columns=['Actual'], index=true.index)
  df_pred['Predicted_delay_one'] = predict
  df_pred['Predicted'] = df_pred['Predicted_delay_one'].shift(1)
  df_pred['Return'] = df_pred['Actual'] - df_pred['Actual'].shift(1)
  return_list = df_pred['Return'].values.tolist()
  position_list = []
  # get position, 1 means long, 0 means short
  for ret in return_list:
    if ret >= 0:
      position_list.append(1)
    if ret < 0:
      position_list.append(0)
  position_list.append(2) # to make the position list has same length of df
  df_pred['delay_position'] = position_list
  df_pred['Position'] = df_pred['delay_position'].shift(1)
  df_pred['Pred_Return'] = df_pred['Predicted_delay_one'] - df_pred['Actual']
  df_pred['Pred_Return'] = df_pred['Pred_Return'].shift(1)
  pred_return_list = df_pred['Pred_Return'].values.tolist()
  pred_pos_list = []
  for ret in pred_return_list:
    if ret >= 0:
      pred_pos_list.append(1)
    if ret < 0:
      pred_pos_list.append(0)
  pred_pos_list.append(2) # to make the position list has same length of df
  df_pred['delay_pred_pos'] = pred_pos_list
  df_pred['Pred_pos'] = df_pred['delay_pred_pos'].shift(1)
  actual_pos_list = df_pred['Position'].values.tolist()[1:]
  predict_pos_list = df_pred['Pred_pos'].values.tolist()[1:]
  actual_return_list = df_pred['Return'].values.tolist()[1:]
  profit_list = []
  accurate_number = 0
  # calculate profit
  for pos, pred_pos, r in zip(actual_pos_list, predict_pos_list, actual_return_list):
    if pos == pred_pos:
      p += abs(r)
      accurate_number += 1
    if pos != pred_pos:
      p -= abs(r)
    profit_list.append(p)
  print("Last {} days, highest profit is ${}, lowest profit is ${}, final profit is ${}".format(TEST_SIZE, max(profit_list), min(profit_list),p))
  # draw diagram
  profit_list.append(2)
  df_pred["Profit"] = profit_list
  df_pred["Profit"] = df_pred["Profit"].shift(1)
  # print(df_pred)
  df_pred[['Actual', 'Predicted']].plot(style='.-')
  plt.title("{} Actual vs Predict Graph".format(name))
  plt.show()

  df_pred["Profit"].plot()
  plt.title("{} Profit and Loss Graph".format(name))
  plt.show()
  
  accurate_rate = accurate_number/len(actual_pos_list)
  print("Last {} days, long short position accurate rate is {}".format(TEST_SIZE, accurate_rate))
  