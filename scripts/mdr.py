#!/usr/bin/python3

"""
mdr.py

This python module contains definitions for multidimensional regression
fitting. Use the mdr.cross_validation to train with cross validation,
or if needed, use mdr.train to train without cross validation.
"""

import math
import numpy as np
import copy

# Compute the gradient of the loss function.
def gradient(loss, pred, x, dx, ins, outs) :
    # Check to see if the input point is a vector or a scalar.
    if hasattr(x, "__getitem__") :
        ys = [pred(i, x) for i in ins]
        l = loss(outs, ys)
        out = [0 for i in range(len(x))]
        for i in range(len(x)) :
            # Use 2 stencil points for better accuracy.
            change1 = [x[j] + (dx[j] if i == j else 0) for j in range(len(x))]
            change2 = [x[j] + (2 * dx[j] if i == j else 0) for j in
                       range(len(x))]
            ys1 = [pred(i, change1) for i in ins]
            ys2 = [pred(i, change2) for i in ins]
            l1 = loss(outs, ys1)
            l2 = loss(outs, ys2)
            out[i] = (2 * l1 - l2 / 2 - 3 * l / 2) / dx[i]
        return np.array(out)
    else :
        # If not a vector, then take the ordinary derivative.
        ys = [pred(i, x) for i in ins]
        l = loss(outs, ys)
        change1 = x + dx
        change2 = x + 2 * dx
        ys1 = [pred(i, change1) for i in ins]
        ys2 = [pred(i, change2) for i in ins]
        l1 = loss(outs, ys1)
        l2 = loss(outs, ys2)
        return (2 * l1 - l2 / 2 - 3 * l / 2) / dx

def train(x, y, pred, loss, w_start, **kwargs):
    #max_iter=100, step=0.01, grad_conv=0.01) :
    """
    Trains the weight vector on a data set.
    x: list of x values. Can be a list of 1d values or a list of vectors.
    y: list of y values.
    pred: function to predict values. Signature is y=pred(x, w)
    loss: Function to find the loss of a certain step. Has the signature
        mle=loss(y, ycalc)
    w_start: The initial guess vector for the weight. Needed to guess the
    dimensions.
    max_iter: The maximum number of steps to take. Defaults to 100
    step: The size of one step. Defaults to 0.01
    grad_conv: the cutoff for the gradient to be considered converged.
    Defaults to 0.01
    print: Whether or not to print output for debugging. Defaults to False.
    Returns the weight vector.
    """

    w = np.array(w_start)
    grad = None
    grad_last = None
    w_last = None
    abs_grad = [100 for i in range(len(w_start))]

    if "max_iter" in kwargs :
        max_iter = kwargs["max_iter"]
    else :
        max_iter = 100

    if "step" in kwargs :
        step = kwargs["step"]
    else :
        step = 0.01

    if "grad_conv" in kwargs :
        grad_conv = kwargs["grad_conv"]
    else :
        grad_conv = 0.01

    if "print" in kwargs :
        printout = kwargs["print"]
    else :
        printout = False
    
    for it in range(max_iter) :
        grad_last = grad
        grad = gradient(loss, pred, w, [step for i in range(len(w_start))],
                        x, y)
        abs_grad = [abs(g) for g in grad]
        if min(abs_grad) < grad_conv :
            if printout :
                print(f"Converged in {it} iterations.")
                print(f"w value: {w}")
            return w
        else :
            # Use the Barzilai-Borwein method to have a variable step size.
            if type(w_last) != type(None) and type(grad_last) != type(None) and\
               all(grad[i] != grad_last[i] for i in range(len(grad))) :
                   stp = step * 100 * abs(np.dot(w - w_last, grad - grad_last))\
                         / np.linalg.norm(grad - grad_last) ** 2
            else :
                stp = step
            w_last = w
            w = w - stp * grad
    raise(RuntimeError(f"Too many iterations!\nw value: {w}"))

def cross_validation(xs, ys, k, pred, loss, w_start, **kwargs) :
    """
    Teach a regression using cross validation.
    xs: List of x values/vectors.
    ys: List of y values.
    k: number of tests.
    pred: regression to teach. Signature of y = pred(x, w)
    loss: Loss function. Signature of mle = loss(y, ycalc)
    w_start: The initial guess vector for the weight.
    max_iter: The maximum number of iterations to run per test. Defaults to 100
    step: The step value for gradient descent. Defaults to 0.01
    grad_conv: The convergence criterion of the gradient. Defaults to 0.01
    print: Whether to print out debugging info. Defaults to False.
    Returns the weight vector.
    """
    # Split the input set into sets to be removed from the training set.
    x_list = np.array_split(xs, k)
    y_list = np.array_split(ys, k)

    w_last = w_start
    for val in range(0, k) :
        # Set up the training set
        train_x = copy.deepcopy(x_list)
        train_y = copy.deepcopy(y_list)

        # Remove values to be excluded.
        val_x = train_x.pop()
        val_y = train_y.pop()

        # Recreate the training set.
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)

        # Train the regression.
        w = train(train_x, train_y, pred, loss, w_last, **kwargs)
        
        # Use the new value as the guess vector.
        # This should be equivalent to taking a weighted average of the results.
        # Source: 
        w_last = w
    return w_last
