import argparse

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn import svm
from scipy.special import expit

import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# sklearn library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Read the CSV file into a DataFrame
# df = pd.read_csv('data.csv')

# Display the first few rows of the DataFrame
# print(df.head())

def generate_random_points(size=10, low=0, high=1):
    data = (high - low) * np.random.random_sample((size, 36)) + low
    return data


def knn_predict(X_train, y_train, x_test, k=1):
    distances = np.sum((X_train - x_test) ** 2, axis=1)
    indices = np.argsort(distances)[:k]
    closest_labels = [y_train[i] for i in indices]
    unique_labels, counts = np.unique(closest_labels, return_counts=True)
    predicted_label = unique_labels[np.argmax(counts)]

    return indices, predicted_label


def sigmoid(z):
    res = expit(z)
    return res


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    res = np.concatenate((intercept, X), axis=1)
    return res


def loss(h, y):
    res = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return res


def learning(X, y, lr=0.01, num_iter=40000):
    X = add_intercept(X)

    # weights initialization
    theta = np.zeros(X.shape[1])

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient

        # if (i % 10000 == 0):
        #     z = np.dot(X, theta)
        #     h = sigmoid(z)
        #     print(f'loss: {loss(h, y)} \t')
        #     # plt.plot(X[y>0.5, 1], X[y>0.5, 2], 'ro', X[y<0.5, 1], X[y<0.5, 2], 'bs')
        #     # # x2 = a*x1 + b
        #     # a = -theta[1] / theta[2]
        #     # b = -theta[0] / theta[2]
        #     # plt.plot(np.array([0, 2]), np.array([0, 2]) * a + b, "g-")
        #     # plt.show()
    return theta


def predict(X, theta, threshold=0.5):
    X = add_intercept(X)
    prob = sigmoid(np.dot(X, theta))
    res = prob >= threshold
    return res


def part_three():
    # fetch dataset
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

    # data (as pandas dataframes)
    X = predict_students_dropout_and_academic_success.data.features
    y = predict_students_dropout_and_academic_success.data.targets

    # Convert feature data and class labels to pandas DataFrames
    X = pd.DataFrame(X, columns=predict_students_dropout_and_academic_success.data.feature_names)
    y = pd.DataFrame(y, columns=["Target"])

    filtered_indices = (y["Target"] == "Dropout") | (y["Target"] == "Graduate")
    other_indices = (y["Target"] == "Enrolled")
    test = X[other_indices]
    X = X[filtered_indices]
    y = y[filtered_indices]

    class_mapping = {"Dropout": 0, "Graduate": 1, "Enrolled": 2}
    y["Target"] = y["Target"].map(class_mapping)

    test = test.values
    X = X.values
    y = y["Target"].values

    X_train = X[:-100]
    X_test = X[-100:]
    y_train = y[:-100]
    y_test = y[-100:]

    #print(X_test)
    #print(y_test)

    indices = np.arange(y.size)
    X = X[indices, :]
    y = y[indices]

    # x_test = np.array([[0.5, 0.5], [1, 1], [1.5, 1.5]])
    # plt.plot(X1[:, 0], X1[:, 1], 'ro', X2[:, 0], X2[:, 1], 'bs')
    # for i in range(3):
    #     plt.plot(x_test[i, 0], x_test[i, 1], 'g^')

    #x_test = np.random.rand(3, 36)  # Generate random test points
    theta = learning(X, y)
    y_pred = predict(test, theta)
    #print(y_pred)

    print("Predicted classes using Logistic Regression Model: ")
    for i in range(100):
        print("Prediction: ", "Positive" if y_pred[i] else "Negative")


    theta = learning(X, y)
    y_pred = predict(X_test, theta)

    a = -theta[1] / theta[2]
    b = -theta[0] / theta[2]
    plt.plot(np.array([0, 2]), np.array([0, 2]) * a + b, "g-")


    print("Predicted classes using Linear Regression Model: ")
    for i in range(100):
        print("Point: ", X_test[i], "Prediction: ", float(y_pred[i]))

    # print("\nPredicted classes using SVM: ")
    # clf = svm.SVC(kernel='linear', C=1000)
    # clf.fit(X, y)
    #
    # predictions = clf.predict(x_test)
    #
    # for i in range(len(predictions)):
    #     print("Point:", x_test[i], "Class prediction:", predictions[i])
    #
    # # plot the decision function
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    #
    # # create grid to evaluate model
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = clf.decision_function(xy).reshape(XX.shape)
    #
    # # plot decision boundary and margins
    # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'])
    # # plot support vectors
    # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k')
    #
    # plt.show()

def final():
    # fetch dataset
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

    # data (as pandas dataframes)
    X = predict_students_dropout_and_academic_success.data.features
    y = predict_students_dropout_and_academic_success.data.targets

    # metadata
    print(predict_students_dropout_and_academic_success.metadata)

    # variable information
    print(predict_students_dropout_and_academic_success.variables)

    y = y['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y = predict_students_dropout_and_academic_success.data.targets

    filtered_indices = (y["Target"] == "Dropout") | (y["Target"] == "Graduate")
    # other_indices = (y["Target"] == "Enrolled")
    # test = X[other_indices]
    # X = X[filtered_indices]
    y = y[filtered_indices]
    #
    class_mapping = {"Dropout": 0, "Graduate": 1, "Enrolled": 2}
    y["Target"] = y["Target"].map(class_mapping)

    #X = predict_students_dropout_and_academic_success.data.features
    X = X[filtered_indices]
    y = y["Target"].values

    print(y)

    # Plotting two features
    feature1_index = 0  # Choose the index of the first feature
    feature2_index = 1  # Choose the index of the second feature

    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.9, wspace=1, hspace=1)
    plt.scatter(X["Age at enrollment"], X["Scholarship holder"], c=y, cmap='viridis')
    plt.title("Scatter Plot of Age of Enrollment vs Scholarship holder")
    plt.colorbar(label='Target')
    # Add custom legend handles and labels to the legend
    plt.xlabel('Age at enrollment')
    plt.ylabel('Scholarship holder')



    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=50000)
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)

    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = logistic_model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='viridis')

    plt.show()

    print("Logistic Regression Accuracy:", logistic_accuracy)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logistic_predictions))

def main():
    #part_one()
    #print()
    #part_two()
    #print()
    #part_three()
    final()


main()
