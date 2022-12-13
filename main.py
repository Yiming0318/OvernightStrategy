'''
Yiming Ge

!!!!!!Notices!!!!!:
yfinace api required!!!!
please pip install yfinance
'''
from regression import*
from constant import*
from classification import*
from deeplearning import*
def main():
    print("############DATA INFO##############")
    # get data and info
    df = get_data(ANALYZE_TICKER, CORRELATED_TICKER, START_DATE, END_DATE, PRICE, FEATURE, ROLLING_WINDOW)
    print_data_info(ANALYZE_TICKER, CORRELATED_TICKER, PRICE, START_DATE, END_DATE, FEATURE, ROLLING_WINDOW)
    df2 = get_data_classfication(ANALYZE_TICKER, CORRELATED_TICKER, START_DATE, END_DATE, PRICE, FEATURE, ROLLING_WINDOW)
    
    # print("############PCA ANALYZER##############")
    # # pca analyze, will show the mean, std, diffenece matrix, eigen value, eigen vector, projected data
    # m, std, D, eig_val, eig_vect, project_data = pca(df, ANALYZE_TICKER, True)
    # print_pca_result(m, std, D, eig_val, eig_vect, project_data)


    print("#############TRAIN TEST SPLIT#############")
    # for regression
    # train test split, will print out percentage and size of train and test
    train1, test1 = my_train_test_split(df, TEST_PERCENT, SEED)
    y_train = train1["Target"]
    y_test = test1["Target"]
    # normalize x matrix
    x_matrix_train = normalize(train1, "Target")
    x_matrix_test = normalize(test1, "Target")
    
    # for classification
    # train test split, will print out percentage and size of train and test
    train2, test2 = my_train_test_split(df2, TEST_PERCENT, SEED)
    y_train2 = train2["Target"]
    y_test2 = test2["Target"]
    # normalize x matrix
    x_matrix_train2 = normalize(train2, "Target")
    x_matrix_test2 = normalize(test2, "Target")

    # for fully connect network
    data = StockDataset(df2)
    print(data[0])
    print(len(data))
    train_loader, test_loader = get_tensor(data)
    test_data = StockDataset(df2[-TEST_SIZE:-1])
    _, test_range_loader = get_tensor(test_data)

    print("#############REGRESSION ANALYZER#############")
    # multilinear regression, will print out the coef and intercept and r_square
    make_multi_linear_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # ridge regression, will print out the coef and intercept and r_square
    make_ridge_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # lasso regression, will print out the coef and intercept and r_square
    make_lasso_regression(x_matrix_train, x_matrix_test, y_train, y_test)
    # random forest tree regressor, will show the first decision tree and print out r_square
    make_random_forest_regressor(x_matrix_train, x_matrix_test, y_train, y_test)
    # k nearest neighbor regressor
    make_knn_regressor(x_matrix_train, x_matrix_test, y_train, y_test)

    print("#############CLASSIFICATION ANALYZER#############")
    # Naive Bayes
    make_naive_bayes(x_matrix_train2, x_matrix_test2, y_train2, y_test2)
    # Random Forest
    make_random_forest(x_matrix_train2, x_matrix_test2, y_train2, y_test2)
    # KNN
    make_knn(x_matrix_train2, x_matrix_test2, y_train2, y_test2)
    # SVC
    make_svm(x_matrix_train2, x_matrix_test2, y_train2, y_test2)

    print("#############FC DEEPNETWORK ANALYZER#############")
    network = FCNet()
    print(network)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    # train the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses, test_counter)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter)
        test(network, test_loader, test_losses, test_counter)
    print('##########')
    # test_given_range(network, test_range_loader, df2, ANALYZE_TICKER)
    # evaluate the performance
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == "__main__":
    main()