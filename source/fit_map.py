"""

Usage:
	fit_map.py from_gt_img_and_corners_file <gt_corners> <gt_img> <floorplan> [--same_scale] [--crop_gt] [--crop_target] [--show] [--debug] 
	fit_map.py from_gt_img <gt_img> <floorplan> [--same_scale] [--crop_gt] [--crop_target] [--show] [--debug]
	fit_map.py from_gt_corners <gt_corners> <floorplan> [--same_scale] [--crop_gt] [--crop_target] [--show] [--debug]
	fit_map.py -h | --help

Options:
	-h --help     Show this screen
"""

import cv2
import json
import time
import math
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from docopt import docopt
import detect_corners
import crop_floorplan


MIN_SEGMENT_LENGTH=20
MIN_AREA_PCTG=0.8
MAX_SCALE_ERROR=0.2


def load_ground_truth_corners(ground_truth_corner_file):
	with open(ground_truth_corner_file, 'r') as f:
		ground_truth_corners = json.load(f) 
	gt_corners = []
	for ind in ground_truth_corners['polylines']:
		polyline = ground_truth_corners['polylines'][ind]
		gt_corners += polyline
	return gt_corners


def gt_img_from_corners(gt_corners_file):
	with open(gt_corners_file, 'r') as f:
		ground_truth_corners = json.load(f) 
	dim0 = 0
	dim1 = 0
	for ind in ground_truth_corners['polylines']:
		polyline = ground_truth_corners['polylines'][ind]
		dim0 = max(dim0, max([v[0] for v in polyline]))
		dim1 = max(dim1, max([v[1] for v in polyline]))

	padding = 5
	dim0 += padding
	dim1 += padding
	dim0 = int(round(dim0))
	dim1 = int(round(dim1))
	gt_img = np.ones((dim1, dim0))*255
	for ind in ground_truth_corners['polylines']:
		polyline = ground_truth_corners['polylines'][ind]
		for ind in range(len(polyline)-1):
			x1, y1 = polyline[ind]
			x2, y2 = polyline[(ind+1)]
			cv2.line(gt_img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 0, 1)

	gt_img = gt_img.astype('uint8')
	gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
	return gt_img


def fit_map_using_corners(gt_corners, gt_img, floorplan, same_scale=False, delta=MIN_SEGMENT_LENGTH, show=False, debug=False):
	if len(gt_img.shape) > 2:
		gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
	else:
		gt_gray = gt_img

	gt_img_cols, gt_img_rows = gt_gray.shape
	gt_img_area = gt_img_rows * gt_img_cols

	# padding is added so that the transformed image of the floorplan can fit into the padded image even in cases
	# where the fit is not correct and results in part of the floorplan being outside of the ground truth floorplan
	padding = max(gt_img_cols, gt_img_rows)
	gt_gray_padded = cv2.copyMakeBorder(gt_gray, padding, padding, padding, padding, borderType=cv2.BORDER_CONSTANT, value=205)
	
	# distance map is used for fast calculation of distance between fitted floorplan boundary points and ground truth
	# boundary points
	dst_map = cv2.distanceTransform(gt_gray_padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	if len(floorplan.shape) > 2:
		floorplan = cv2.cvtColor(floorplan, cv2.COLOR_BGR2GRAY)
	floorplan_cols, floorplan_rows = floorplan.shape
	floorplan_area = floorplan_cols * floorplan_rows

	# extracting locations of boundary points from the floorplan
	boundary_indices = np.array(np.nonzero(floorplan==0))
	boundary_indices = np.matrix.transpose(boundary_indices)

	# converting boundary indices into homogeneous coordinates
	boundary_indices_h = np.c_[ boundary_indices, np.ones(len(boundary_indices)) ]
	
	corners = detect_corners.detect_corners(floorplan)
	print('Number of detected corners in the floorplan: ', len(corners))

	# preselect line segments above certain length for use in the fitting process
	num_corners = len(corners)
	corner_pairs = []
	for k in range(num_corners):
		for l in range(num_corners):
			fpt1 = np.array(corners[k])
			fpt2 = np.array(corners[l])
			d = detect_corners.calculate_distance(fpt1, fpt2)
			if d > delta:
				corner_pairs.append([k, l, d])

	num_gt_corners = len(gt_corners)
	min_fitness_score = float('Inf')
	fitted_indices_0 = None
	fitted_indices_1 = None

	if debug:
		print('Number of corner pairs: ', len(corner_pairs))

	# In this loop we pick any two corners from the ground truth and try to match them to two corners
	# in the floorplan; 
	# We calculate the transformation that transforms the corners from the floorplan into the points in the 
	# ground truth map and then evaluate how well that transformation brings the floorplan to the ground
	# truth.
	# The best fitting transformation is selected as the final result.
	for i in tqdm(range(num_gt_corners)):
		pt1 = np.array(gt_corners[i])
		
		for j in range(i+1, num_gt_corners):
			pt2 = np.array(gt_corners[j])
			gt_d = detect_corners.calculate_distance(pt1, pt2)

			if gt_d < delta:
				continue

			for ind in range(len(corner_pairs)):
				fpt1 = np.array(corners[corner_pairs[ind][0]])
				fpt2 = np.array(corners[corner_pairs[ind][1]])
				d = corner_pairs[ind][2]

				scale_factor = gt_d/d

				if same_scale and (scale_factor > 1 + MAX_SCALE_ERROR or scale_factor < 1 - MAX_SCALE_ERROR):
					continue 

				resized_floorplan_area = floorplan_area*scale_factor*scale_factor

				# If after fitting the floorplan is too small, then skip evaluation, it is not the right transform.
				# This reduces search space.
				ratio = resized_floorplan_area/gt_img_area
				if ratio <= MIN_AREA_PCTG:
					continue

				height, width = floorplan.shape

				fpt1_scaled = scale_factor * fpt1
				fpt2_scaled = scale_factor * fpt2

				# find the angle that the floorplan segment between corners needs to be rotated by to coincide with the selected 
				# ground truth line segment between the selected corners
				angle = ang([pt1, pt2], [fpt1, fpt2])

				fpt2_scaled_translated = fpt2_scaled + pt1 - fpt1_scaled

				# arccos only gives result from 0 to 180 degrees; this helps us refine the angle between the two line segments
				if (fpt2_scaled_translated[0]-pt1[0]) * (pt2[1] - pt1[1]) - (fpt2_scaled_translated[1] - pt1[1])*(pt2[0]-pt1[0]) < 0:
					angle = -angle

				# constructing scaling matrix
				Ms = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
				# getting rotation matrix 
				Mr = cv2.getRotationMatrix2D((fpt1_scaled[1], fpt1_scaled[0]), angle, 1)
				Mr = np.matrix.transpose(Mr)

				# combining scaling and rotation into a single transformation
				Msr = np.matmul(Ms, Mr)
				current_fitted_indices = np.matmul(boundary_indices_h, Msr)

				current_fitted_indices = current_fitted_indices.astype('int64')
				current_fitted_indices = np.matrix.transpose(current_fitted_indices)
				
				# translation can be applied through matrix operations too, but this turned out
				# to be faster
				t = pt1 - fpt1_scaled
				translated_indices_0 =  current_fitted_indices[0] + int(round(t[1])) + padding
				translated_indices_1 =  current_fitted_indices[1] + int(round(t[0])) + padding

				if len(translated_indices_0) == 0:
					continue

				fitness_score = evaluate_fit(dst_map, translated_indices_0, translated_indices_1)

				if min_fitness_score > fitness_score:

					fitted_indices_0 = translated_indices_0.copy()
					fitted_indices_1 = translated_indices_1.copy()

					min_fitness_score = fitness_score

	if debug:
		print('min_fitness_score = ', min_fitness_score)

	if fitted_indices_0 is not None:
		fitted_floorplan = np.ones(gt_gray_padded.shape)*255

		# Creating the transformed floorplan image
		for i in range(len(fitted_indices_0)):
			cv2.circle(fitted_floorplan, (fitted_indices_1[i], fitted_indices_0[i]), 1, (0, 0, 0), -1)

		gt_fitness_score, max_dist_gt = evaluate_reverse_fit(fitted_indices_0, fitted_indices_1, gt_gray_padded)

		if debug:
			print('ground truth fitness score: ', gt_fitness_score)
			print('max_dist_gt = ', max_dist_gt)

		if show:
			# this shows the transformed floorplan overlaid on the ground truth
			fitted_map_img = cv2.cvtColor(gt_gray_padded, cv2.COLOR_GRAY2BGR)
			for i in range(len(fitted_indices_0)):
				cv2.circle(fitted_map_img, (fitted_indices_1[i], fitted_indices_0[i]), 1, (125, 125, 125), -1)
			cv2.imshow('fitted map image', fitted_map_img)
			cv2.waitKey(0)

		metrics = {'fitness_score': min_fitness_score, 'gt_fitness_score': gt_fitness_score, 'max_dist_gt': max_dist_gt}
		fitted_boundary_indices = [fitted_indices_0, fitted_indices_1]
		return metrics, fitted_floorplan, fitted_boundary_indices


def evaluate_fit(gt_distance_map, indices0, indices1):
	try:
		dist_sum = np.sum(gt_distance_map[indices0, indices1])
	except:
		return float('Inf')
	count = len(indices0)

	return dist_sum/count


def evaluate_reverse_fit(indices0, indices1, gt):
	img = np.ones(gt.shape) * 255
	img[indices0.astype('int64'), indices1.astype('int64')] = 0
	img = img.astype('uint8')

	dist_map = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	dist_sum = np.sum(dist_map[gt==0])
	max_dist = np.max(dist_map[gt==0])

	nonzero_indices = np.nonzero(gt==0)
	count = len(nonzero_indices[0])

	return dist_sum/count, max_dist


# the code below is from internet https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2/28261304
# and only slightly modified to make sure proper domain of values for acos function, also small naming changes

def dot(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]


def ang(line1, line2):
    # Get nicer vector form
    v1 = [(line1[0][0]-line1[1][0]), (line1[0][1]-line1[1][1])]
    v2 = [(line2[0][0]-line2[1][0]), (line2[0][1]-line2[1][1])]
    # Get dot prod
    dot_prod = dot(v1, v2)
    # Get magnitudes
    mag1 = dot(v1, v1)**0.5
    mag2 = dot(v2, v2)**0.5
    # Get cosine value
    cos_ = dot_prod/mag1/mag2
    # Get angle in radians and then convert to degrees

    if cos_ > 1:
    	cos_ = 1
    elif cos_ < -1:
    	cos_ = -1
    angle = math.acos(cos_)

    return math.degrees(angle)		


if __name__ == "__main__":
	arguments = docopt(__doc__)
	if arguments['from_gt_img_and_corners_file']:
		gt_corners = load_ground_truth_corners(arguments['<gt_corners>'])
		gt_img = cv2.imread(arguments['<gt_img>'])
	elif arguments['from_gt_img']:
		gt_img = cv2.imread(arguments['<gt_img>'])
		if len(gt_img.shape) > 2:
			gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
		show_corners = False
		gt_corners = detect_corners.detect_corners(gt_img, show=arguments['--debug'])
	elif arguments['from_gt_corners']:
		gt_corners = load_ground_truth_corners(arguments['<gt_corners>'])
		gt_img = gt_img_from_corners(arguments['<gt_corners>'])

		if arguments['--debug']:
			cv2.imshow('gt_img', gt_img)
			cv2.waitKey(0)
	else:
		exit("{0} is not a command. See 'options.py --help'.".format(arguments['<command>']))

	floorplan = arguments['<floorplan>']
	same_scale=False
	if arguments['--same_scale']:
		same_scale = True
	show=False
	if arguments['--show']:
		show = True
	debug=False
	if arguments['--debug']:
		debug = True

	floorplan = cv2.imread(floorplan)
	if arguments['--crop_target']:
		floorplan = crop_floorplan.crop_floorplan(floorplan)
	if arguments['--crop_gt']:
		gt_img = crop_floorplan.crop_floorplan(gt_img)
	fit_map_using_corners(gt_corners=gt_corners, gt_img=gt_img, floorplan=floorplan, same_scale=same_scale, show=show, debug=debug)


			
