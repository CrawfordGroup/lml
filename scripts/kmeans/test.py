#!/usr/bin/python3

"""
test.py

usage: test.py [-h] [--2d {on,off}] [--3d {on,off}] [--print-debug {on,off}]

Test the k-means algorithm against various random sets.

optional arguments:
  -h, --help            show this help message and exit
  --2d {on,off}         Whether to show the graph of the 2D clustering.
  --3d {on,off}         Whether to show the graph of the 3D clustering.
  --incomplete {on,off}
                        Whether to show the graph showing the result when the
                        k-means algorithm returns a value that has fewer than
                        k clusters.
  --clustered {on,off}  Whether to show the graph of the pre-clustered points.
  --print-debug {on,off}
                        Whether to print the debug information.

"""
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import kmeans
import argparse

def main(**kwargs) :
    """
    Tests the k-means algorithm using random points and spits out a graph if
    desired.
    """
    printout = (kwargs["print_debug"] == 'on')

    points1 = [[rand.randrange(0, 10), rand.randrange(0, 10)]
                   for i in range(10)]
    points2 = [[rand.randrange(0, 10), rand.randrange(0, 10),
                rand.randrange(0, 10)] for i in range(50)]
    # When trained, this will cause only two clusters to be formed.
    points3 = [[9, 0], [3, 1], [4, 4], [8, 8], [7, 0], [6, 8], [6, 9], [5, 3],
               [9, 0], [7, 0]]
    # Designed to have 3 clusters.
    points4 = []
    expected4 = []
    start_center4 = [[rand.randint(0, 10), rand.randint(0, 10)]
                     for i in range(3)]
    for i in range(3) :
        expected4.append([])
        for j in range(10) :
            point = [start_center4[i][0] + (2 * rand.random() - 1) / 2,
                            start_center4[i][1] + (2 * rand.random() - 1) / 2]
            expected4[i].append(point)
            points4.append(point)

    # Calculate the clusters and scores of each.
    center1, cluster1, score1, _, _, _ = \
                kmeans.find_best(points1, 3, printout = printout,
                                full_return = True)
    center2, cluster2, score2, _, _, _ = \
                kmeans.find_best(points2, 3, printout = printout,
                                full_return = True)
    center3, cluster3 = kmeans.train(points3, 3,
                                initial_points=[[9, 0], [7, 0], [3, 1]])
    center4, cluster4, score4, _, _, _ = \
             kmeans.find_best(points4, 3, printout=printout, full_return = True)

    figure = plt.figure()
    plot_n = 1
    plot_total = sum(1 for i in kwargs
                     if i in ["2d", "3d", "incomplete", "clustered"]\
                     and kwargs[i] == "on")
    # Find how many rows and columns of subplots to show.
    plot_r = round(math.sqrt(plot_total))
    if plot_total != 0 :
        plot_c = math.ceil(plot_total / plot_r)
    colors = ["r", "g", "b"]
    colors2 = ["c", "m", "y"]

    # Show the first plot.
    if kwargs["2d"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        plot_n += 1
        ax.set_title(
            f"First Test: 2D, three clusters, 10 points. Score = {score1}")
        for i in range(len(cluster1)) :
            ax.scatter([r[0] for r in cluster1[i]],
                       [r[1] for r in cluster1[i]],
                       color = colors[i], marker = 'o', label=f"Cluster {i}")
            ax.scatter([center1[i][0]], [center1[i][1]], color = colors[i],
                       marker = "s", label=f"Center of cluster {i}")
        ax.legend()

    # Show the second plot.
    if kwargs["3d"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n, projection="3d")
        plot_n += 1
        ax.set_title(
            f"Second Test: 3D, three clusters, 50 points. Score = {score2}")
        for i in range(len(cluster2)) :
            ax.scatter3D([r[0] for r in cluster2[i]],
                         [r[1] for r in cluster2[i]],
                         [r[2] for r in cluster2[i]], color = colors[i],
                         marker = 'o')
            ax.scatter3D([center2[i][0]], [center2[i][1]], [center2[i][2]],
                         color = colors[i], marker = 's')
    # Show the third plot.
    if kwargs["incomplete"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        plot_n += 1
        ax.set_title(
            "Third Test: 2D, 3 clusters expected, 2 returned. " +\
            f"Score = {kmeans.score_dist(cluster3, center3, 3)}")
        for i in range(len(cluster3)) :
            ax.scatter([r[0] for r in cluster3[i]],
                        [r[1] for r in cluster3[i]],
                        color = colors[i], marker = 'o', label=f"Cluster {i}")
            ax.scatter([center3[i][0]],
                       [center3[i][1]], color=colors[i], marker = 's',
                       label=f"Center of cluster {i}")
        ax.legend()
    # Show the fourth plot.
    if kwargs["clustered"] == "on" :
        ax = figure.add_subplot(plot_r, plot_c, plot_n)
        plot_n += 1
        ax.set_title(
            f"Fourth test: 2D, 30 strongly clustered points. Score = {score4}")
        for i in range(len(cluster4)) :
            ax.scatter([r[0] for r in cluster4[i]],
                       [r[1] for r in cluster4[i]],
                       color = colors[i], marker = 'o', label=f"Cluster {i}")
            ax.scatter([center4[i][0]], [center4[i][1]],
                       color = colors[i], marker = 's')
        for i in range(len(expected4)) :
            ax.scatter([r[0] for r in expected4[i]],
                       [r[1] for r in expected4[i]],
                       color = colors2[i], marker='+',
                       label=f"Expected groupings for cluster {i}")
            ax.scatter([start_center4[i][0]], [start_center4[i][1]],
                       color = colors2[i], marker = 'x')
        ax.legend()
    if plot_total != 0 :
        plt.show()
    

# Set up the argument parser.
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
        description="Test the k-means algorithm against various random sets.")
    parser.add_argument("--2d",
                        help="Whether to show the graph of the 2D clustering.",
                        default="on", type=lambda s : str(s).lower(),
                        choices=["on", "off"],
                        required=False)
    parser.add_argument("--3d",
                        help="Whether to show the graph of the 3D clustering.",
                        default="on", type=lambda s : str(s).lower(),
                        choices=["on", "off"],
                        required=False)
    parser.add_argument("--incomplete",
                        help="Whether to show the graph showing the result when the k-means algorithm returns a value that has fewer than k clusters.",
                        default="on", type=lambda s : str(s).lower(),
                        choices=["on", "off"],
                        required=False)
    parser.add_argument("--clustered",
                        help="Whether to show the graph of the pre-clustered points.",
                        default="on", type=lambda s : str(s).lower(),
                        choices=["on", "off"],
                        required=False)
    parser.add_argument("--print-debug",
                        help="Whether to print the debug information.",
                        default="on", type=lambda s : str(s).lower(),
                        choices=["on", "off"],
                        required=False)
    main(**vars(parser.parse_args()))
