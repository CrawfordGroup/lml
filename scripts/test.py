#!/usr/bin/python3

"""
test.py

This script tests the mdr.py module and displays graphs of each
test regression. The types of graphs displayed can be changed with
command-line arguments.

"""

import mdr
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
from matplotlib import cm
import argparse
import random as rand

# Standard root-mean-squared loss function.
def l2_loss(y, ycalc) :
    ls = [(y[i] - ycalc[i]) ** 2 for i in range(len(y))]
    mle = math.sqrt(sum(ls)) / len(ls)
    return mle

# Sum of the squares loss function.
def l2_square(y, ycalc) :
    ls = [(y[i] - ycalc[i]) ** 2 for i in range(len(y))]
    mle = sum(ls) / len(ls)
    return mle

# Calculate the r-squared correlation coefficient.
def r2(y, ycalc) :
    avg = sum(y) / len(y)
    sum_res = sum((y[i] - ycalc[i]) ** 2 for i in range(len(y)))
    sum_tot = sum((y[i] - avg) ** 2 for i in range(len(y)))
    return (1 - sum_res / sum_tot)

def main(**kwargs) :
    input0 = list(range(10))
    input1 = list(range(0, 10))
    input2 = [[i // 10, i % 10] for i in range(0, 100)]
    input3 = list(range(10))
    input4 = [[i // 10, i % 10] for i in range(0, 100)]

    output0 = [i if i < 7 else 2 * i for i in input0]
    output1 = [5 * i + 1 +
               2 * (2 * rand.random() - 1) for i in input1]
    output2 = [x[0] + 2 * x[1] + 3 +
               2 * (2 * rand.random() - 1) for x in input2]
    output3 = [2 * x ** 2 - x + 1 +
               10 * (2 * rand.random() - 1) for x in input3]
    output4 = [5 * (x[0] - 1) ** 2 + 2 * (x[1] + 3) ** 2 +
               10 * (2 * rand.random() - 1) for x in input4]

    pred0 = lambda x, w : w[0] * x
    pred1 = lambda x, w : w[0] * x + w[1]
    pred2 = lambda x, w : w[0] * x[0] + w[1] * x[1] + w[2]
    pred3 = lambda x, w : w[0] * x ** 2 + w[1] * x + w[2]
    pred4 = lambda x, w : w[0] * (x[0] - w[1]) ** 2 + \
            w[2] * (x[1] - w[3]) ** 2

    plot_n = 1
    plot_total = 0

    # Find out how many plots will be shown.
    if kwargs["lin_1d"] != "off" :
        plot_total += 1
    if kwargs["lin_1d_bias"] != "off" :
        plot_total += 1
    if kwargs["lin_2d"] != "off":
        plot_total += 1
    if kwargs["non_lin_1d"] != "off" :
        plot_total += 1
    if kwargs["non_lin_2d"] != "off" :
        plot_total += 1

    # Find how many rows and columns of subplots to show.
    plot_r = 2 if plot_total > 3 else 1
    plot_c = math.ceil(plot_total / plot_r)
    figure = plt.figure()

    # Whether or not to print debug information.
    if kwargs["print_debug"] == "on" :
        print("Test case 0:")
        w0 = mdr.cross_validation(input0, output0, 3, pred0, l2_square,
                                  [0], printout=True)
        print(f"r² = {r2(output0, [pred0(i, w0) for i in input0])}")

        print("\nTest case 1:")
        w1 = mdr.cross_validation(input1, output1, 3, pred1, l2_square,
                                  [0, 0], printout=True)
        print(f"r² = {r2(output1, [pred1(i, w1) for i in input1])}")

        print("\nTest case 2:")
        w2 = mdr.cross_validation(input2, output2, 3, pred2, l2_square,
                                  [0, 0, 0], printout=True)
        print(f"r² = {r2(output2, [pred2(i, w2) for i in input2])}")

        print("\nTest case 3:")
        w3 = mdr.cross_validation(input3, output3, 10, pred3, l2_square,
                                  [0, 0, 0], printout=True)
        print(f"r² = {r2(output3, [pred3(i, w3) for i in input3])}")

        print("\nTest case 4:")
        w4 = mdr.cross_validation(input4, output4, 10, pred4, l2_loss,
                                  [1, 0, 1, 0], printout=True)
        print(f"r² = {r2(output4, [pred4(i, w4) for i in input4])}")

    else :
        w0 = mdr.cross_validation(input0, output0, 3, pred0, l2_square,
                                  [0])
        w1 = mdr.cross_validation(input1, output1, 3, pred1, l2_square,
                                  [0, 0])
        w2 = mdr.cross_validation(input2, output2, 3, pred2, l2_square,
                                  [0, 0, 0])
        w3 = mdr.cross_validation(input3, output3, 10, pred3, l2_square,
                                  [0, 0, 0])
        w4 = mdr.cross_validation(input4, output4, 10, pred4, l2_loss,
                                  [1, 0, 1, 0])

    # Show the first test plot.
    if kwargs["lin_1d"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        ax.scatter(input0, output0, label="Expected", color='r', marker='o')
        ax.plot(input0, [pred0(i, w0) for i in input0],
                 label = f"Predicted r² = {r2(output0, [pred0(i, w0) for i in input0])}")
        ax.set_title("First Test: 1-D Linear, Zero Intercept")
        ax.legend()
        plot_n += 1

    # Show the second test plot.
    if kwargs["lin_1d_bias"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        ax.scatter(input1, output1, label="Expected", color='r', marker='o')
        ax.plot(input1, [pred1(i, w1) for i in input1],
                 label=f"Predicted r² = {r2(output1, [pred1(i, w1) for i in input1])}")
        ax.set_title("Second Test: 1-D Linear")
        ax.legend()
        plot_n += 1

    # Show the third plot
    X = list(range(10))
    Y = list(range(10))
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[pred2([X[i][j], Y[i][j]], w2) for j in range(len(X[i]))]
        for i in range(len(X))])
    if kwargs["lin_2d"] == "surface" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n, projection="3d")
        ax.scatter3D([i[0] for i in input2], [i[1] for i in input2], output2,
                 depthshade = False, color='r', marker = 'o')
        ax.plot_surface(X, Y, Z, alpha = 0.5)
        ax.contourf(X, Y, Z, zdir="z", offset=-1, cmap=cm.coolwarm)
        ax.contourf(X, Y, Z, zdir="x", offset=-1, cmap=cm.coolwarm)
        ax.contourf(X, Y, Z, zdir="y", offset=11, cmap=cm.coolwarm)
        ax.set_title(f"Third Test: 2-D Linear r² = {r2(output2, [pred2(i, w2) for i in input2])}")
        plot_n += 1
    elif kwargs["lin_2d"] == "contour" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        ax.contourf(X, Y, Z, cmap=cm.coolwarm)
        ax.set_title(f"Third Test: 2-D Linear r² = {r2(output2, [pred2(i, w2) for i in input2])}")
        plot_n += 1

    # Show the fourth plot
    if kwargs["non_lin_1d"] == "on":
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        ys = [pred3(x, w3) for x in input3]
        ax.scatter(input3, output3, label="Expected", color='r', marker='o')
        ax.plot(input3, ys, label=f"Predicted r² = {r2(output3, ys)}")
        ax.set_title("Fourth Test: 1-D Non-linear")
        ax.legend()
        plot_n += 1

    # Show the fifth plot
    X = list(range(10))
    Y = list(range(10))
    Xs = [i[0] for i in input4]
    Ys = [i[1] for i in input4]
    ys = [pred4(i, w4) for i in input4]
    Xm, Ym = np.meshgrid(X, Y)
    Z = np.array([[pred4([Xm[i][j], Ym[i][j]], w4) for j in range(len(Xm[i]))]
                  for i in range(len(Xm))])
    if kwargs["non_lin_2d"] == "surface" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n, projection="3d")
        ax.scatter3D(Xs, Ys, output4, depthshade = False, color='r', marker='o')
        ax.plot_surface(Xm, Ym, Z, alpha = 0.5)

        ax.contourf(Xm, Ym, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
        ax.contourf(Xm, Ym, Z, zdir='x', offset=-1, cmap=cm.coolwarm)
        ax.contourf(Xm, Ym, Z, zdir='y', offset=11, cmap=cm.coolwarm)
    
        ax.set_title(f"Fifth Test: 2-D Non-linear r² = {r2(output4, ys)}")
        plot_n += 1
    elif kwargs["non_lin_2d"] == "contour" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        ax.contourf(X, Y, Z, cmap = cm.coolwarm)
        ax.title(f"Fifth Test: 2-D Non-linear r² = {r2(output4, ys)}")
        plot_n += 1

    plt.show()

if __name__ == "__main__" :
    # Set up command line arguments.
    parser = argparse.ArgumentParser(
        description="Test the regression learning algorithm against various regression types.")

    parser.add_argument("--lin-1d", help="Whether to show the graph of the 1D linear regression with no intercept.",
                        default="on", type=str, choices=["on", "off"], required = False)
    parser.add_argument("--lin-1d-bias", help="Whether to show the graph of the 1D linear regression with intercept.",
                        default="on", type=str, choices=["on", "off"], required = False)
    parser.add_argument("--lin-2d", help="Whether to show the graph of the 2D linear regression with intercept.",
                        default="surface", type=str, choices=["surface", "contour", "off"], required = False)
    parser.add_argument("--non-lin-1d", help="Whether to show the graph of the 1D non-linear regression.",
                        default="on", type=str, choices=["on", "off"], required = False)
    parser.add_argument("--non-lin-2d", help="Whether to show the graph of the 2D non-linear regression.",
                        default="surface", type=str, choices=["surface", "contour", "off"], required = False)
    parser.add_argument("--print-debug", help="Whether to print debug information.",
                        default="on", type=str, choices=["on", "off"], required=False)

    main(**vars(parser.parse_args()))
