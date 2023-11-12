import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


colors = ["blue", "green"]


def generate_linear_points(n):
    train_points = []
    test_points = []

    for i in range(int(n / 2)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(50, 100), random.randint(50, 100), 0))
        else:
            train_points.append(Point(random.randint(50, 100), random.randint(50, 100), 0))

    for i in range(int(n / 2)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(0, 50), random.randint(0, 50), 1))
        else:
            train_points.append(Point(random.randint(0, 50), random.randint(0, 50), 1))

    for point in train_points:
        plt.scatter(point.x, point.y, color=colors[point.color])
    return train_points, test_points


def generate_non_linear_points(n):
    train_points = []
    test_points = []
    for i in range(int(n / 2)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(40, 60), random.randint(40, 60), 0))
        else:
            train_points.append(Point(random.randint(40, 60), random.randint(40, 60), 0))

    for i in range(int(n / 4)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(0, 20), random.randint(0, 20), 1))
        else:
            train_points.append(Point(random.randint(0, 20), random.randint(0, 20), 1))

    for i in range(int(n / 4)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(80, 100), random.randint(80, 100), 1))
        else:
            train_points.append(Point(random.randint(80, 100), random.randint(80, 100), 1))
    for point in train_points:
        plt.scatter(point.x, point.y, color=colors[point.color])
    return train_points, test_points


def svm_alg(model, train_points, test_points):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(train_points)):
        x_train.append([train_points[i].x, train_points[i].y])
        y_train.append((train_points[i].color))
    for i in range(len(test_points)):
        x_test.append([test_points[i].x, test_points[i].y])
        y_test.append((test_points[i].color))

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    X = x_train
    y = y_train
    X = np.array(X)
    y = np.array(y)
    return model


def plot_svc_decision_function(model, ax=None, plot_support=True):
    #Лимиты для графика
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #Одномерные массивы чисел с равномерно распределенными значениями в заданном диапазоне
    x = np.linspace(xlim[0], xlim[1], 10)
    y = np.linspace(ylim[0], ylim[1], 10)

    #Созданием сетки точек в двумерном пространстве
    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T

    #Решающая функция
    P = model.decision_function(xy).reshape(X.shape)

    #levels=[-1, 0, 1] - это расстояния от гиперплоскости
    ax.contour(X, Y, P, colors='black',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

def choice(isLianear):
    if (isLianear == True):
        model = SVC(kernel='linear', C=1E6)
        train_points, test_points = generate_linear_points(100)
    else:
        model = SVC(kernel='rbf', C=1E6)
        train_points, test_points = generate_non_linear_points(100)
    return model, train_points, test_points

if __name__ == "__main__":
    isLinear = True

    model, train_points, test_points = choice(isLinear)

    plt.savefig('figure1')

    model = svm_alg(model, train_points, test_points)
    plot_svc_decision_function(model)
    plt.savefig('figure2')

    point = [90, 88]

    new_point_class = model.predict([point])[0]
    print("The new point belongs to class " + str(new_point_class) + " and is colored " + colors[
        new_point_class])

    plt.scatter(point[0], point[1], color=colors[new_point_class])
    plt.savefig('figure3')
