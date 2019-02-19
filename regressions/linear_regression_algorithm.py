from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i+1 for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs)) ** 2 - mean(xs ** 2)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    print(y_mean_line)
    plt.plot(xs, y_mean_line, color='y')
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_line = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_line)


xs, ys = create_dataset(40, 80, 2, correlation=False)
m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]
print(regression_line)
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

predict_x = xs[len(xs)-1] + 10
predict_y = (m * predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line, color='b')
plt.show()
