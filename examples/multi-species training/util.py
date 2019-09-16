import mlqm
import numpy as np
import matplotlib.pyplot as plt
import json
import math

def average(data) :
    """
    Find the average value of a data set.
    """
    
    if hasattr(data, "__len__") :
        return sum(data) / len(data)
    else :
        total = 0
        num = 0
        for d in data :
            total += d
            num += 1
        return total / num

def pochhammer(x, n) :
    prod = 1
    if n == 0 :
        return 1
    if n < 0  :
        return 0
    for i in range(n) :
        prod *= x + i
    return prod

def median(data) :
    """
    Find the median of a data set.
    """
    srt = sorted(data)
    if len(srt) % 2 == 0 :
        return (srt[len(srt) / 2] + srt[len(srt) / 2 - 1]) / 2
    else :
        return srt[len(srt) / 2]

def sample_metric(data, order, avg = None) :
    """
    Find metrics of a sample data set.
    If avg == None, calculates the average again.
    This option is to allow the user to pre-calculate and
    pass the average, if wanted, to speed up calculations.
    """
    if avg is None :
        avg = average(data)
    return sum((d - avg) ** order for d in data) / len(data)

def population_metric(data, order, avg = None) :
    """
    Find metrics of a population data set.
    If avg == None, calculates the average again.
    This option is to allow the user to pre-calculate and
    pass the average, if wanted, to speed up calculations.
    """
    if avg is None :
        avg = average(data)
    return sum((d - avg) ** order for d in data) / pochhammer(len(data), order)

def covariance(data1, data2, avg1 = None, avg2 = None) :
    """
    Find the covariance between two data sets. Like with
    sample and population metric functions, averages may be
    pre-calculated and passed in.
    """
    assert(len(data1) == len(data2))
    if avg1 == None :
        avg1 = average(data1)
    if avg2 == None :
        avg2 = average(data2)
    
    return sum((data1[i] - avg1) * (data2[i] - avg2)
               for i in range(len(data1))) / len(data1)

def residues(data, avg = None) :
    """
    Find the residues of a set from its average.
    Average may be pre-calculated and passed.
    Returns the list of residues, followed by the average
    absolute residue.
    """
    if avg == None :
        avg = average(data)

    return [d - avg for d in data], sum(abs(d - avg)
                                        for d in data) / len(data)

def difference_metrics(data1, data2, order) :
    """
    Find expressions of the form
    sum((xi - yi) ** k for i) / n
    """

    assert(len(data1) == len(data2))
    return sum((data1[i] - data2[i]) ** order for i
               in range(len(data1))) / len(data1)

def beta(a, b) :
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

def beta(x, a, b) :
    """
    Solve the de
    y' = t ^ (a - 1) (1 - t) ^ (b - 1)
    for t = 0 to x.
    """
    startx = 0
    steps = 1000
    y = 0
    f = lambda t: t ** (a - 1) * (1 - t) ** (b - 1)
    st = (x - startx) / steps
    f0 = f(0)
    f1 = f(st / 2)
    f2 = f(st)

    for i in range(steps) :
        f0 = f2
        f1 = f(st * (i + 1/2))
        f2 = f(st * (i + 1))
        y += st * (f0 + 4 * f1 + f2) / 6
    return y

def t_value(expectation, df) :
    return 1 - (beta(df / (expectation ** 2 + df),
                     df / 2, 1 / 2) / beta(df / 2, 1 / 2)) / 2

def r_squared(pred, calc, pred_avg = None) :
    if pred_avg is None :
        pred_avg = average(pred)
    return 1 - sum((pred[i] - calc[i]) ** 2 for i in range(len(pred))) /\
           sum((pred[i] - pred_avg) ** 2 for i in range(len(pred)))
