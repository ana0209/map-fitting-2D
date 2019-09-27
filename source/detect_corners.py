import numpy as np
import math
import cv2

# These are constants input into cv2.HoughLinesP; for more explanation see opencv documentation
MIN_LINE_LENGTH=10
MAX_LINE_GAP=3

# This constant is used when detecting corners. A corner is detected if two line segments are roughly
# perpendicular and if they have end points (one from each line segment) that are closer than MAX_CORNER_GAP
MAX_CORNER_GAP=5

def detect_line_segments(floorplan, min_line_length=MIN_LINE_LENGTH, max_line_gap=MAX_LINE_GAP, show_detected_segments=False):

	# edges are occupied cells whose value is zero, the rest is either free or unknown space
	floorplan_edges = np.zeros(floorplan.shape)

	# creating an image with edges having 255 values because that is the format that works
	# with cv2.HoughLinesP
	for row in range(floorplan.shape[1]):
		for col in range(floorplan.shape[0]):
			if floorplan[col][row] == 0:
				floorplan_edges[col][row] = 255

	floorplan_edges = floorplan_edges.astype('uint8')

	line_segments = cv2.HoughLinesP(floorplan_edges,1,np.pi/180, 10, min_line_length, max_line_gap)

	line_segments = [line_segment[0] for line_segment in line_segments]

	if show_detected_segments:
		floorplan = cv2.cvtColor(floorplan, cv2.COLOR_GRAY2BGR)
		floorplan = floorplan.copy()
		for x1, y1, x2, y2 in line_segments:
			cv2.line(floorplan,(x1,y1),(x2,y2),(0,0,255),2)
		cv2.imshow('Detected segments', floorplan)
		cv2.waitKey(0)
		      
	return line_segments


def calculate_distance(p1,p2):
	x1, y1 = p1
	x2, y2 = p2
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
	return dist  


def get_angle(line_segment):
	x1, y1, x2, y2 = line_segment
	dx = x2 - x1
	dy = y2 - y1
	rads = math.atan2(-dy, dx)
	return math.degrees(rads)


def detect_corners_from_line_segments(line_segments, delta=MAX_CORNER_GAP, debug=False):

	angles = []
	for line_segment in line_segments:
		angles.append(get_angle(line_segment))

	corners = []
	for i in range(len(line_segments)):
		for j in range(i+1, len(line_segments)):
			# If line segments are at an angle; It should be 90 degree 
			# angle, but we give a wide buffer to account for inaccuracies
			# in calculated segments; 
			# It is only important that they are not parallel or close
			# to parallel because then they do not make a corner;
			if 45 < abs(angles[i] - angles[j]) < 135:
				x1, y1, x2, y2 = line_segments[i]
				p11 = np.array([x1, y1])
				p12 = np.array([x2, y2])
				x1, y1, x2, y2 = line_segments[j]
				p21 = np.array([x1, y1])
				p22 = np.array([x2, y2])

				if calculate_distance(p11, p21) < delta:
					corners.append((p11 + p21)/2.0)
				elif calculate_distance(p11, p22) < delta: 
					corners.append((p11 + p22)/2.0)
				elif calculate_distance(p12, p21) < delta:
					corners.append((p12 + p21)/2.0)
				elif calculate_distance(p12, p22) < delta:
					corners.append((p12 + p22)/2.0)

	if debug:
		print('Number of detected corners is ', len(corners), '.')

	return corners


def detect_corners(floorplan, max_corner_gap=MAX_CORNER_GAP, min_line_length=MIN_LINE_LENGTH, max_line_gap=MAX_LINE_GAP, show=False):
	line_segments = detect_line_segments(floorplan, min_line_length=min_line_length, max_line_gap=max_line_gap, show_detected_segments=show)
	corners = detect_corners_from_line_segments(line_segments, delta=max_corner_gap)

	if show:
		floorplan_color = cv2.cvtColor(floorplan, cv2.COLOR_GRAY2BGR)
		for corner in corners:
			corner = corner.astype('uint16')
			cv2.circle(floorplan_color, (corner[0], corner[1]), 3, (0, 0, 255), -1)

		cv2.imshow('corners', floorplan_color)
		cv2.waitKey(0)

	return corners

