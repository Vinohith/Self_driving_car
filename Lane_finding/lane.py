import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Line:
	def __init__(self, x1, y1, x2, y2):
		self.x1 = np.float32(x1)
		self.y1 = np.float32(y1)
		self.x2 = np.float32(x2)
		self.y2 = np.float32(y2)
		self.slope = self.compute_slope()
		self.bias = self.compute_bias()
	def compute_slope(self):
		return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)
	def compute_bias(self):
		return (self.y1 - self.slope*self.x1)
	def get_coord(self):
		return np.array([self.x1, self.y1, self.x2, self.y2])
	def draw(self, img, color = [255,0,0], thickness = 5):
		cv2.line(img, (self.x1,self.y1), (self.x2,self.y2), color, thickness)



def roi():
	pass

def pipline(img):
	pass

test_images = [os.path.join('test_images', name) for name in os.listdir('test_images')]
for test_image in test_images:
	img = cv2.imread(test_image, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)
	img_edge = cv2.Canny(img_blur, threshold1 = 50, threshold2 = 80)
	# cv2.imshow('edges', img_edge)
	# cv2.waitKey(0)

	lines = cv2.HoughLinesP(img_edge, 2, np.pi/180, 1, 15, 5)

	print(lines.shape)
	print(type(lines))

	lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
	candinate_lines = []
	for line in lines:
		if 0.5 <= np.abs(line.slope) <= 2:
			candinate_lines.append(line)

	pos_lines = [l for l in candinate_lines if l.slope>0]
	print(len(pos_lines))
	neg_lines = [l for l in candinate_lines if l.slope<0]
	print(len(neg_lines))


	neg_bias = np.int32(np.median([l.bias for l in neg_lines]))
	print(neg_bias)
	neg_slope = np.median([l.slope for l in neg_lines])
	print(neg_slope)
	x1, y1 = 0, neg_bias
	x2, y2 =  -np.int32(neg_bias/neg_slope), 0
	left_lane = Line(x1, y1, x2, y2)



	pos_bias = np.int32(np.median([l.bias for l in pos_lines]))
	print(pos_bias)
	pos_slope = np.median([l.slope for l in pos_lines])
	print(pos_slope)
	x1, y1 = 0, pos_bias
	x2, y2 = img_gray.shape[1], np.int32(pos_slope*img_gray.shape[1] + pos_bias)
	# x2, y2 = np.int32(np.round((img_gray.shape[0] - pos_bias) / pos_slope)), img_gray.shape[0]
	print(x1, y1, x2, y2)
	right_lane = Line(x1, y1, x2, y2)


	line_img = np.zeros((img.shape[0], img.shape[1]))
	print(line_img.shape)
	left_lane.draw(line_img)
	right_lane.draw(line_img)
	# cv2.imshow('fr', line_img)
	# cv2.waitKey(0)


	mask = np.zeros_like(line_img)
	print(mask.shape)
	cv2.fillPoly(mask, np.array([[(450,320), (490,320), (910,540), (50,540)]]), 255)
	masked_img = cv2.bitwise_and(line_img, mask)
	# cv2.imshow('fr', mask)
	# cv2.waitKey(0)






	# img_line = np.copy(img)*0

	# for line in lines:
	#     for x1,y1,x2,y2 in line:
	#         cv2.line(img_line,(x1,y1),(x2,y2),(255,0,0),10)
	masked_img = np.uint8(masked_img)
	color_edges = np.dstack((masked_img, np.zeros_like(masked_img), np.zeros_like(masked_img)))

	lines_edges = cv2.addWeighted(color_edges, 0.8, img, 1, 0) 

	output = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB)

	cv2.imshow('output', output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()




	# mask = np.zeros_like(img_edge)
	# cv2.fillPoly(mask, np.array([[(450,310), (490,310), (910,540), (50,540)]]), 255)
	# masked_img = cv2.bitwise_and(img_edge, mask)

	 

	# cv2.imshow('blur', img_blur)
	# cv2.waitKey(0)
	# cv2.imshow('edge', img_edge)
	# cv2.waitKey(0)
	# cv2.imshow('mask', masked_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imshow('edges', lines_edges)
	# cv2.waitKey(0)
	# # cv2.imshow('mask', masked_img)
	# # cv2.waitKey(0)
	# cv2.destroyAllWindows()