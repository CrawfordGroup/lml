#!/bin/python3

# Multi-dimensional machine learning.

import math
import numpy as np
import random as rand
import matplotlib as mpl
import copy

def gradient(loss, pred, x, dx, ins, outs) :
    if hasattr(x, "__getitem__") :
        ys = [pred(i, x) for i in ins]
        l = loss(outs, ys)[1]
        out = [0 for i in range(len(x))]
        for i in range(len(x)) :
            # Use 2 stencil points for better accuracy.
            change1 = [x[j] + (dx[j] if i == j else 0) for j in range(len(x))]
            change2 = [x[j] + (2 * dx[j] if i == j else 0) for j in range(len(x))]
            ys1 = [pred(i, change1) for i in ins]
            ys2 = [pred(i, change2) for i in ins]
            l1 = loss(outs, ys1)[1]
            l2 = loss(outs, ys2)[1]
            out[i] = (2 * l1 - l2 / 2 - 3 * l / 2) / dx[i]
        
        return np.array(out)            
    else :
        ys = [pred(i, x) for i in ins]
        l = loss(outs, ys)[1]
        change1 = x + dx
        change2 = x + 2 * dx
        ys1 = [pred(i, change1) for i in ins]
        ys2 = [pred(i, change2) for i in ins]
        l1 = loss(outs, ys1)[1]
        l2 = loss(outs, ys2)[1]
        return (2 * l1 - l2 / 2 - 3 * l / 2) / dx

def train(x, y, pred, loss, w_start, max_iter=100, step=0.01, grad_conv=0.01) :
    """
    Trains the weight vector on a data set.
    x: list of x values. Can be a list of 1d values or a list of vectors.
    y: list of y values.
    pred: function to predict values. Signature is y=pred(x, w)
    loss: Function to find the loss of a certain step. Has the signature
        ls, mle=loss(y, ycalc)
    w_start: The initial guess for the weight. Needed to guess the dimensions.
    max_iter: The maximum number of steps to take.
    step: The size of one step.
    grad_conv: the cutoff for the gradient to be considered converged.
    """

    w = np.array(w_start)
    w_list = [w]
    print(x)
    y_p_list = []

    mse_list = []
    abs_grad = [100 for i in range(len(w_start))]

    for it in range(max_iter) :
        y_p = [pred(x_i, w) for x_i in x]
        l, mse = loss(y, y_p)
        mse_list.append(mse)
        grad = gradient(loss, pred, w, [step for i in range(len(w_start))], x, y)
        abs_grad = [np.linalg.norm(g) for g in grad]
        if min(abs_grad) < grad_conv :
            print("Converged in %d iterations."%(it))
            print("w value: {}".format(w))
            print("mse: {}".format(mse))
            return y_p_list, w, mse
        else :
            w = w - step * grad
            w_list.append(w)
    raise(RuntimeError("Too many iterations!\nw value: {}\nmse: {}".format(w, mse)))

def cross_validation(xs, ys, k, pred, loss, params) :
    """
    Teach a regression using cross validation.
    xs: List of x values/vectors.
    ys: List of y values.
    k: number of tests.
    pred: regression to teach. Signature of y = pred(x, w)
    loss: Loss function. Signature of l, mse = loss(y, ycalc)
    params: Number of dimensions of the weight vector.
    """
    x_list = np.array_split(xs, k)
    y_list = np.array_split(ys, k)

    y_p_list = []
    mse_list = []
    w_list = []
    w_last = [0 for i in range(params)]
    w_list.append(w_last)
    for val in range(0, k) :
        train_x = copy.deepcopy(x_list)
        train_y = copy.deepcopy(y_list)
        val_x = train_x.pop()
        val_y = train_y.pop()
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        y, w, mse = train(train_x, train_y, pred, loss, w_last, 1000, 0.001)
        y_p = [pred(val_x_i, w) for val_x_i in val_x]
        l, mse = loss(val_y, y_p)
        y_p_list.append(y_p)
        mse_list.append(mse)
        w_list.append(w)
        w_last = w
    return np.mean(mse_list), y_p_list, mse_list, w_last, w_list
    
