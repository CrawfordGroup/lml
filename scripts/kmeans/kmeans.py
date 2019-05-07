#!/usr/bin/python3

"""
kmeans.py

Find clusters of points using the k-means algorithm.
"""

import numpy as np
import random as rand
import math

# Pick a random sequence without any duplicates.
def random_seq(points, choices) :
    hold = points.copy()
    out = []
    for i in range(choices) :
        rnd = rand.choice(hold)
        # Filter out duplicates.
        hold = list(filter(lambda x : x != rnd, hold))
        out.append(rnd)
    return out

def score_dist(clusters, centers, k) :
    """
    Score a configuration of clusters with the square root of the sum of the
    variances.
    """
    pts = np.array(clusters)
    cts = np.array(centers)
    variance = [sum(np.linalg.norm(pts[i][j] - cts[i]) ** 2 for j in range(len(pts[i])))
              for i in range(len(cts))]
    return math.sqrt(sum(variance))

# Find if two clusters are equal under permutation.
def equal_under_changes(c1, c2, depth = None) :
    """
    Finds if two lists are equal under permutation.
    The depth parameter tells the function whether to check for
    equality on lists within lists or to stop at a certain level,
    comparing lists, like coordinates, as values not equal under
    permutations. Returns true if the two lists are permutations of
    each other, false if otherwise.
    """
    if depth != None :
        if not hasattr(c1[0], "__len__") or depth == 0 :
            return all(c1[i] in c2 for i in range(len(c1)))
        else :
            return all(any(equal_under_changes(c1[i], c2[k], depth - 1)
                           for k in range(len(c2))) for i in range(len(c1)))
    if not hasattr(c1[0], "__len__") :
        return all(c1[i] in c2 for i in range(len(c1)))
    else :
        return all(any(equal_under_changes(c1[i], c2[k])
                       for k in range(len(c2))) for i in range(len(c1)))
   
def train(points, k, **kwargs) :
    """
    Trains a k-means clustering problem.
    points: List of points to test.
    k: Number of clusters.
    initial_points: List of points to use as a seed for the algorithm. If
    None, use random points from the list. If fewer than k points,
    fill with random points until there are k points (default None).
    max_iters: The maximum number of iterations to use (default 100).
    printout: Whether or not to print the results (default False).
    Returns the centers for the clusters, then the clusters themselves.
    """
    # Store points so that we can see if we are in a cycle.
    pts = []
    last_pts = []
    prev_centers = []
    # If we have initial points, use them. If we don't, then don't.
    if "initial_points" in kwargs :
        if len(kwargs["initial_points"]) < k :
            pts = kwargs["initial_points"]
            rest = [p for p in points if pts.count(p) == 0]
            pts.append(random_seq(rest, k - len(pts)))
        else :
            pts = kwargs["initial_points"][0:k]
    else :
        pts = random_seq(points, k)

    start = pts
    
    if "max_iters" not in kwargs :
        max_iters = 100
    else :
        max_iters = kwargs["max_iters"]

    for its in range(max_iters) :
        last_pts = pts
        # Set up the clusters to be empty.
        clusters = [[] for i in range(k)]
        for p in points :
            # Sort centers by nearest.
            cls = sorted(list(range(len(pts))),
                         key = lambda i :
                         np.linalg.norm(np.array(p) - np.array(pts[i])))
            # Add point to the nearest cluster.
            clusters[cls[0]].append(p)
            
        # Recalculate the centers, filtering out empty clusters, so as to
        # not choke the algorithm.
        clusters = list(filter(lambda x : x != [], clusters))
        pts = [sum(np.array(p) for p in cluster) / len(cluster)
               for cluster in clusters if len(cluster) != 0]

        if any(equal_under_changes(pts, prev) for prev in prev_centers[:-2]) :
            if "printout" in kwargs and kwargs["printout"]:
                print(f"Cycled in {its} iterations.")
                print(f"Centers of the clusters are {pts}.")
                print(f"Clusters are {clusters}")
            return pts, clusters
        prev_centers.append(pts)
        if equal_under_changes(pts, last_pts) :
            if "printout" in kwargs and kwargs["printout"]:
                print(f"Converged in {its} iterations.")
                print(f"Centers of the clusters are {pts}.")
                print(f"Clusters are {clusters}")
            return pts, clusters
    if "printout" in kwargs and kwargs["printout"] :
        print(points)
    raise RuntimeError("Too many iterations!")

def find_best(points, k, **kwargs) :
    """
    Finds the best (or almost best) configuration by trying multiple
    starting conditions.

    points: The points to cluster.
    k: The number of clusters.
    printout: Whether to print the results. Also passed to train (default False).
    tests: The number of tests to do (default 10).
    full_return: Whether to return an extended amount of information. The new
    returns will be centers, clusters, score of the clustering,
    previous centers, previous clusters, previous scores.
    one_return: If true, only return the centers of the clusters, rather than
    the centers and clusters.
    All others are passed to train.
    Returns the centers and the clusters of the best configuration.
    """
    # Store each test case so that we can find the best.
    configs = []
    centers = []
    scores = []
    seeds = []

    if list(kwargs.keys()).count("tests") == 0 :
        max_its = 10
    else :
        max_its = kwargs["tests"]

    for i in range(max_its) :
        seed = random_seq(points, k)
        # Check to see if we have already tried this seed. This 10 can be
        # changed. 
        for i in range(10) :
            if any(equal_under_changes(seed, s, 0) for s in seeds) :
                # If we have already tried it, try another one.
                seed = random_seq(points, k)
        # Otherwise, just use the seed.
        seeds.append(seed)

        center, clusters = train(points, k, initial_points = seed, **kwargs)
        
        # If we already have this result, just ignore it and continue on.
        if any(equal_under_changes(clusters, c, 1) for c in configs) :
            continue
        configs.append(clusters)
        centers.append(center)

        scores.append(score_dist(clusters, center, k))
        if "printout" in kwargs and kwargs["printout"] :
            print(f"Score: {scores[-1]}\n")
    # Find the best configuration and return it.
    min_score = scores.index(min(scores))

    if "printout" in kwargs and kwargs["printout"] :
        print(f"Best clusters: {configs[min_score]}")
        print(f"Best centers: {centers[min_score]}")
        print(f"Best score: {scores[min_score]}\n")
    
    if "full_return" in kwargs and kwargs["full_return"] :
        return centers[min_score], configs[min_score], min(scores), centers, \
               configs, scores
    if "one_return" in kwargs and kwargs["one_return"] :
        return centers[min_score]
    return centers[min_score], configs[min_score]
