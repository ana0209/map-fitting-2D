import cv2
import numpy as np

PADDING = 5

def crop_floorplan(floorplan, padding=PADDING, low_t=0, high_t=255, show=False):
	if len(floorplan.shape) > 2:
		floorplan = cv2.cvtColor(floorplan, cv2.COLOR_BGR2GRAY)

	rows = floorplan.shape[1]
	cols = floorplan.shape[0]

	min_col = floorplan.shape[0]
	max_col = -1

	for row in range(rows):
		tmp = floorplan[:][row]
		tmpind1=np.where(tmp <= 0)
		tmpind2=np.where(tmp >= 255)

		if len(tmpind1[0])==0:
			tmpind1=[cols+1,-1]
		else:
			tmpind1=tmpind1[0]
		if len(tmpind2[0])==0:
			tmpind2=[cols+1,-1]
		else:
			tmpind2=tmpind2[0]

		min_col=min(min_col, min(tmpind1[0], tmpind2[0]))
		max_col=max(max_col, max(tmpind1[-1], tmpind2[-1]))

	min_row = rows
	max_row = -1

	for col in range(cols):
		tmp = floorplan[:][col]
		tmpind1=np.where(tmp <= 0)
		tmpind2=np.where(tmp >= 255)

		if len(tmpind1[0])==0:
			tmpind1=[rows+1,-1]
		else:
			tmpind1=tmpind1[0]
		if len(tmpind2[0])==0:
			tmpind2=[rows+1,-1]
		else:
			tmpind2=tmpind2[0]

		min_row=min(min_row, min(tmpind1[0], tmpind2[0]))
		max_row=max(max_row, max(tmpind1[-1], tmpind2[-1]))


	top = max(0, min_col - padding)
	bottom = min(floorplan.shape[0], max_col + padding)

	left = max(0, min_row - padding)
	right = min(floorplan.shape[1], max_row + padding)

	if show:
		cv2.imshow('image', floorplan)
		cv2.imshow('cropped image', floorplan[top:bottom+1, left:right+1])
		cv2.waitKey(0)

	return floorplan[top:bottom+1, left:right+1]

