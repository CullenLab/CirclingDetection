# -*- coding: utf-8 -*-
# Circling Mouse Utilities
#   Use for common fcns across methods
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, expon
from scipy.optimize import curve_fit

############################################################################
# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

def orientation(p, q, r):
	# to find the orientation of an ordered triplet (p,q,r)
	# function returns the following values:
	# 0 : Colinear points
	# 1 : Clockwise points
	# 2 : Counterclockwise
	
	# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
	# for details of below formula.
	
	val = float(q[1] - p[1]) * (r[0] - q[0]) - float(q[0] - p[0]) * (r[1] - q[1])
	if (val > 0):
		# Clockwise orientation
		return 1
	elif (val < 0):
		# Counterclockwise orientation
		return 2
	else:
		# Colinear orientation
		return 0

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
	if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
		(q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1])) ):
		return True
	return False

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
	# Find the 4 orientations required for
	# the general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if ((o1 != o2) and (o3 != o4)):
		return True

	# Special Cases
	# p1 , q1 and p2 are colinear and p2 lies on segment p1q1
	if ((o1 == 0) and onSegment(p1, p2, q1)):
		return True
	# p1 , q1 and q2 are colinear and q2 lies on segment p1q1
	if ((o2 == 0) and onSegment(p1, q2, q1)):
		return True
	# p2 , q2 and p1 are colinear and p1 lies on segment p2q2
	if ((o3 == 0) and onSegment(p2, p1, q2)):
		return True
	# p2 , q2 and q1 are colinear and q1 lies on segment p2q2
	if ((o4 == 0) and onSegment(p2, q1, q2)):
		return True
	# If none of the cases
	return False
	
# This code is contributed by Ansh Riyal
##########################################################################

# from https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
def minBoundingRect(points):
    """
    Find the smallest (by area) bounding box for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, np.pi/2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-np.pi/2),
        np.cos(angles+np.pi/2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    bbox_points = np.zeros((4, 2))
    bbox_points[0] = np.dot([x1, y2], r)
    bbox_points[1] = np.dot([x2, y2], r)
    bbox_points[2] = np.dot([x2, y1], r)
    bbox_points[3] = np.dot([x1, y1], r)
    
    width = abs(x1-x2)
    height = abs(y1-y2)
    aspect = max([width/height, height/width])
    diameter = max(width, height)

    return aspect, diameter

def checkAngle(nturns):
    body_vector_angles = nturns - nturns[0]
    
    # Instead of ONLY taking final versus start, we actually want to look at things frame-by-frame
    # to prevent messing up where we turn > 180 degrees
    angle_deltas = np.diff(body_vector_angles)
    
    # total angle as fraction of a full circle; convert radians to circles
    num_turns = np.abs(np.sum(angle_deltas) / (2*np.pi))
    
    return num_turns

def checkDuplicates(x, y):
    for point1 in range(1, len(x)):
        for point2 in range(point1+3, len(x)):
            if doIntersect((x[point1], y[point1]), (x[point1+1], y[point1+1]), (x[point2-1], y[point2-1]), (x[point2], y[point2])):
                return True
    return False
    
# function for fitting 1d distributions
def poisson_func(x,
                 w_p,b_p):
    return w_p*np.exp(-b_p*x)

def guassian_func(x,
                  w_g,mu_g,sigma_g):
    return w_g*np.exp(-(x-mu_g)**2/2/sigma_g**2)

def combined_poisson_guassian_func(x,
                                   w_p,b_p,
                                   w_g,mu_g,sigma_g):
    return w_p*np.exp(-b_p*x) + w_g*np.exp(-(x-mu_g)**2/2/sigma_g**2)

def fit_Turns_dist(this_X):
    # kernel guassian est
    kernel = gaussian_kde(this_X)
    
    # we will fit the kernel guassian distribution with combined poisson and guassian dist
    X_point = np.arange(0,2.5,0.001)
    popt, pcov = curve_fit(combined_poisson_guassian_func, X_point, kernel.evaluate(X_point),
                           bounds=((0,0,0,0.5,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
    return popt

def fit_Minor_dist(this_X):
    # kernel guassian est
    kernel = gaussian_kde(this_X)
    
    # we will fit the kernel guassian distribution with combined poisson and guassian dist
    X_point = np.arange(0,3,0.001)
    popt, pcov = curve_fit(combined_poisson_guassian_func, X_point, kernel.evaluate(X_point),
                           bounds=((0,0,0,0.5,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
    return popt

def fit_Major_dist(this_X):
    # kernel guassian est
    kernel = gaussian_kde(this_X)
    
    # we will fit the kernel guassian distribution with combined poisson and guassian dist
    X_point = np.arange(0,3,0.001)
    popt, pcov = curve_fit(combined_poisson_guassian_func, X_point, kernel.evaluate(X_point),
                           bounds=((0,0,0,1,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
    return popt