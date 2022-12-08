'''
Yiming Ge
Stock regression
Run this file to get the regression results
including:
Multi linear regression
Ridge regression
Lasso regresiion
Random forest tree regressor
!!!!!!Notices!!!!!:
yfinace api required!!!!
please pip install yfinance
'''
from utility_function import*
from constant import*

def main():
    print("############DATA INFO##############")
    # get data and info
    df = get_data(ANALYZE_TICKER, CORRELATED_TICKER, START_DATE, END_DATE, PRICE, FEATURE, ROLLING_WINDOW)
    print_data_info(ANALYZE_TICKER, CORRELATED_TICKER, PRICE, START_DATE, END_DATE, FEATURE, ROLLING_WINDOW)
 
    
    # print("############PCA ANALYZER##############")
    # # pca analyze, will show the mean, std, diffenece matrix, eigen value, eigen vector, projected data
    # m, std, D, eig_val, eig_vect, project_data = pca(df, ANALYZE_TICKER, True)
    # print_pca_result(m, std, D, eig_val, eig_vect, project_data)


    print("#############TRAIN TEST SPLIT#############")
    # train test split, will print out percentage and size of train and test
    train, test = my_train_test_split(df, TEST_PERCENT, SEED)
    y_train = train[ANALYZE_TICKER]
    y_test = test[ANALYZE_TICKER]
    # normalize x matrix
    x_matrix_train = normalize(train, ANALYZE_TICKER)
    x_matrix_test = normalize(test, ANALYZE_TICKER)

    print("#############REGRESSION ANALYZER#############")
    # multilinear regression, will print out the coef and intercept and r_square
    make_multi_linear_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # # ridge regression, will print out the coef and intercept and r_square
    # make_ridge_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # lasso regression, will print out the coef and intercept and r_square
    make_lasso_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # random forest tree regressor, will show the first decision tree and print out r_square
    make_random_forest_regressor(x_matrix_train, x_matrix_test, y_train, y_test)
    # k nearest neighbor regressor
    make_knn_regressor(x_matrix_train, x_matrix_test, y_train, y_test)
    # support vector machine regressor
    make_svm_regressor(x_matrix_train, x_matrix_test, y_train, y_test)


if __name__ == "__main__":
    main()