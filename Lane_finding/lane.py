# importing the necessary packages
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque


# Defining the Line class
class Line:
	'''
	Line defined by (x1,y1) and (x2,y2).
	Slope = (y2 - y1) / (x2 - x1)
	Bias = (y1 - slope * x1)
	y - y1 = [(y2 - y1) / (x2 - x1)] * (x - x1)
	'''

	# Initializing the Line when the object is created 
	def __init__(self, x1, y1, x2, y2):
		self.x1 = np.float32(x1)
		self.y1 = np.float32(y1)
		self.x2 = np.float32(x2)
		self.y2 = np.float32(y2)
		self.slope = self.compute_slope()
		self.bias = self.compute_bias()

	# Computing the line slope
	def compute_slope(self):
		return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

	# Computing the line bias
	def compute_bias(self):
		return (self.y1 - self.slope*self.x1)

	# returning the line coordinates
	def get_coord(self):
		return np.array([self.x1, self.y1, self.x2, self.y2])

	# function to draw the line
	def draw(self, img, color = [255,0,0], thickness = 10):
		cv2.line(img, (self.x1,self.y1), (self.x2,self.y2), color, thickness)



def region_of_interest(edge_image, vertices):
	'''
	Applies a mask on the desired region of image 
	defined by the vertices
	:param edge_image: input image
	:param vertices: desired pixel polygon
	'''

	# defining a dark mask
	mask = np.zeros_like(edge_image)
	# filling the desired pixels within the vertices as white
	cv2.fillPoly(mask, vertices, 255)
	# applying the mask to extract to desied portion in the image
	cropped_image = cv2.bitwise_and(mask, edge_image)
	return cropped_image


def image_processing(image, image_height, image_width):
	'''
	Applying some basic pre-processing on the image
	:param image: input BGR image
	:param image_height: height of image (540)
	:param image_width: width of image (960)
	'''

	# converting the input BGR image to Gray
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Applying a gaussian blur to smoothen the image
	# so that the next Canny filter extracts better 
	# edges due to reduction of noise 
	blur_image = cv2.GaussianBlur(gray_image, (17,17), 0)
	# Detecting the edges 
	edge_image = cv2.Canny(blur_image, threshold1=50, threshold2=100)
	return edge_image



def get_lane_lines(image, image_height, image_width, vertices):
	'''
	This function tries to predict the lane lines in the image
	:param image: input BGR image
	:param image_height: height of image (540)
	:param image_width: width of image (960)
	:param vertices: desired pixel polygon
	'''

	# output after performing basic pre-processing on the image
	edge_image = image_processing(image, image_height, image_width)

	# extracting the required portion of the image
	cropped_image = region_of_interest(edge_image, vertices)

	# generating all the points present in the cropped portion 
	# of the image (in our case along the desired lane) by 
	# performing hough transform
	detected_lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=5, lines=np.array([]), 
									 minLineLength=15, maxLineGap=25)

	# convert the (x1, y1, x2, y2) into lines by initializing 
	# the Line object with its coordinates, slope and bias
	detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

	# keeping only those lines whose slope is between 
	# 30 degrees and 60 degrees
	candinate_lines = []
	for line in detected_lines:
		if 0.5 <= np.abs(line.slope) <= 5:
			candinate_lines.append(line)
 
	# separating the lines with +ve and -ve slopes
	pos_lines = [line for line in candinate_lines if line.slope>0]
	neg_lines = [line for line in candinate_lines if line.slope<0]

	# computing the line the best fits the left lane
	left_lane_slope = np.median([line.slope for line in neg_lines])
	left_lane_bias = np.median([line.bias for line in neg_lines])
	x1, y1 = 0, left_lane_bias
	x2, y2 =  -np.int32(left_lane_bias / left_lane_slope), 0
	left_lane = Line(x1, y1, x2, y2)

	# computing the line the best fits the right lane
	right_lane_slope = np.median([line.slope for line in pos_lines])
	right_lane_bias = np.median([line.bias for line in pos_lines])
	x1, y1 = 0, right_lane_bias
	x2, y2 = image_width, np.int32(right_lane_slope*image_width + right_lane_bias)
	right_lane = Line(x1, y1, x2, y2)

	return left_lane, right_lane


# https://github.com/ndrplz/self-driving-car/blob/1eadaca5e39b0448385db7ac2de0732d6dd7e600/project_1_lane_finding_basic/lane_detection.py#L133
def smoothen_over_time(lane_lines):
	"""
	Smooth the lane line inference over a window of frames and returns the average lines.
	:param lane_lines: list of infered lines over a window of frames
	"""

	avg_line_lt = np.zeros((len(lane_lines), 4))
	avg_line_rt = np.zeros((len(lane_lines), 4))

	for t in range(0, len(lane_lines)):
	    avg_line_lt[t] += lane_lines[t][0].get_coord()
	    avg_line_rt[t] += lane_lines[t][1].get_coord()

	return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))



def lane_detection_pipline(frame):
	'''
	Start of the lane detection pipeline. Takes a list of frames as input and outputs 
	an image overlaid with infered lane lines. 
	:param frame: input list of frames (RGB)
	'''

	# checks if it is a video os single image
	is_videoclip = len(frame)>0
	image_height, image_width = frame[0].shape[0], frame[0].shape[1]
	
	# defined vertices of region of interest
	vertices = np.array([[(0,image_height), (450,320), (510,320), (image_width,image_height)]], dtype=np.int32)

	# infering the lane lines for each frame and smoothening over a window
	lane_lines = []
	for t in range(0, len(frame)):
		infered_lines = get_lane_lines(frame[t], image_height, image_width, vertices)
		lane_lines.append(infered_lines)
	lane_lines = smoothen_over_time(lane_lines)

	# defining a dark background to draw the final lane line which will later 
	# be overlaid on the original image after extracting the requied region
	line_image = np.zeros((image_height, image_width))
	# drawing the left and right lanes on line_image
	for lane in lane_lines:
		lane.draw(line_image)

	# extracting the required region of interest from the image which has only
	# the final left and right lanes infered 
	line_image = region_of_interest(line_image, vertices)

	line_image = np.uint8(line_image)
	# stacking along the z-axis to create a cloured line_image
	color_line_image = np.dstack((line_image, np.zeros_like(line_image), np.zeros_like(line_image)))

	# if the input is a video then the original image is the last input but if the
	# input is a single image the oiginal image is the image itself
	original_image = frame[-1] if is_videoclip else frame[0]

	# the aoutput will the weighted addition of the line_image (which has
	# the infered left and right lanes) and the original image
	output = cv2.addWeighted(color_line_image, 0.8, original_image, 1, 0)
	return output





if __name__ == '__main__':

	# for images
	test_images = [os.path.join('test_images', name) for name in os.listdir('test_images')]
	for image in test_images:
		out_path = os.path.join('out', 'images', os.path.basename(image))
		input_image = cv2.cvtColor(cv2.imread(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		output_image = lane_detection_pipline([input_image])
		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_path, output_image)
		cv2.imshow('output', output_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# for videos
	test_videos = [os.path.join('test_videos', name) for name in os.listdir('test_videos')]
	for video in test_videos:
		cap = cv2.VideoCapture(video)
		out_path = os.path.join('out', 'video', os.path.basename(video))
		out = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), 
							  fps=20.0, frameSize=(960, 540))
		frame_buffer = deque(maxlen=10)
		while cap.isOpened():
			ret, color_frame = cap.read()
			if ret:
				color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
				# color_frame = cv2.resize(color_frame, (540, 960))
				frame_buffer.append(color_frame)
				blend_frame = lane_detection_pipline(frame_buffer)
				out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
				cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
			else:
				break
		cap.release()
		out.release()
		cv2.destroyAllWindows()
