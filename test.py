#!/bin/python3

import mdml
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle as marker
from mpl_toolkits.mplot3d import Axes3D as plt3d

def l2_loss(y, ycalc) :
    ls = [(y[i] - ycalc[i]) ** 2 for i in range(len(y))]
    mle = sum(ls) / len(ls)
    return ls, mle

def r2(y, ycalc) :
    avg = sum(y) / len(y)
    sum_res = sum((y[i] - ycalc[i]) ** 2 for i in range(len(y)))
    sum_tot = sum((y[i] - avg) ** 2 for i in range(len(y)))
    return (1 - sum_res / sum_tot)

def main() :
    input0 = list(range(10))
    input1 = list(range(0, 10))
    input2 = [[i // 10, i % 10] for i in range(0, 100)]
    output0 = [i for i in input0]
    output1 = [5 * i + 1 for i in input1]
    output2 = [x[0] + 2 * x[1] + 3 for x in input2]
    pred0 = lambda x, w : w[0] * x
    pred1 = lambda x, w : w[0] * x + w[1]
    pred2 = lambda x, w : w[0] * x[0] + w[1] * x[1] + w[2]

    _, _, _, w0, _ = mdml.cross_validation(input0, output0, 3, pred0, l2_loss, 1)
    _, _, _, w1, _ = mdml.cross_validation(input1, output1, 3, pred1, l2_loss, 2)
    _, w2, _ = mdml.train(input2, output2, pred2, l2_loss, [0, 0, 0])
    # Show the zeroth test plot.
    plt.figure()
    plt.scatter(input0, output0, label="Expected", color='r', marker='o')
    plt.plot(input0, [pred0(i, w0) for i in input0],
             label = f"Predicted r² = {r2(output0, [pred0(i, w0) for i in input0])}")
    plt.title("Zeroth Test: 1-D Linear, Zero Intercept")
    plt.legend()

    # Show the first test plot.
    plt.figure()
    plt.scatter(input1, output1, label="Expected", color='r', marker=marker('o'))
    plt.plot(input1, [pred1(i, w1) for i in input1],
             label=f"Predicted r² = {r2(output1, [pred1(i, w1) for i in input1])}")
    plt.title("First Test: 1-D Linear")
    plt.legend()

    # Show the Second plot
    X = [i[0] for i in input2]
    Y = [i[1] for i in input2]
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[pred2([X[i][j], Y[i][j]], w2) for j in range(len(X[i]))]
         for i in range(len(X))])
    plt.figure()
    ax = plt.gca(projection="3d")
    ax.scatter3D([i[0] for i in input2], [i[1] for i in input2], output2,
                 depthshade = False, color='r', marker = 'o')
    ax.plot_surface(X, Y, Z)
    plt.title(f"Third Test: 2-D Linear r² = {r2(output2, [pred2(i, w2) for i in input2])}")

    plt.show()

def __init__() :
    import importlib
    importlib.reload(mdml)

if __name__ == "__main__" :
    main()
