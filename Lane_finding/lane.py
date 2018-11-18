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
	def draw(self, img, color = [255,0,0], thickness = 10):
		cv2.line(img, (self.x1,self.y1), (self.x2,self.y2), color, thickness)



def region_of_interest(edge_image, vertices):
	mask = np.zeros_like(edge_image)
	cv2.fillPoly(mask, vertices, 255)
	cropped_image = cv2.bitwise_and(mask, edge_image)
	return cropped_image


def image_processing(image, image_height, image_width):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur_image = cv2.GaussianBlur(gray_image, (17,17), 0)
	edge_image = cv2.Canny(blur_image, threshold1=50, threshold2=100)
	return edge_image



def lane_detection_pipline(image):
	image_height, image_width = image.shape[0], image.shape[1]
	edge_image = image_processing(image, image_height, image_width)
	# cv2.imshow('edge_image', edge_image)
	# cv2.waitKey(0)

	# vertices = np.array([[(0,image_height), (image_width/2,image_height/2), (image_width,image_height)]], dtype=np.int32)
	vertices = np.array([[(0,image_height), (450,320), (510,320), (image_width,image_height)]], dtype=np.int32)
	cropped_image = region_of_interest(edge_image, vertices)
	# cv2.imshow('cropped_image', cropped_image)
	# cv2.waitKey(0)

	# detected_lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/180, threshold=160, lines=np.array([]), 
	# 								 minLineLength=40, maxLineGap=25)
	detected = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=5, lines=np.array([]), 
									 minLineLength=40, maxLineGap=25)
	detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected]
	candinate_lines = []
	for line in detected_lines:
		if 0.5 <= np.abs(line.slope) <= 5:
			candinate_lines.append(line)

	pos_lines = [line for line in candinate_lines if line.slope>0]
	neg_lines = [line for line in candinate_lines if line.slope<0]


	# for line in detected:
	#     for x1,y1,x2,y2 in line:
	#         cv2.line(cropped_image,(x1,y1),(x2,y2),(255,0,0),10)

	# cv2.imshow('img', cropped_image)
	# cv2.waitKey(0)

	# print(len(candinate_lines))

	# for l in pos_lines:
	# 	print(l.slope)

	# for l in neg_lines:
	# 	print(l.slope)

	left_lane_slope = np.median([line.slope for line in neg_lines])
	# print(left_lane_slope)
	left_lane_bias = np.median([line.bias for line in neg_lines])
	# print(left_lane_bias)
	x1, y1 = 0, left_lane_bias
	x2, y2 =  -np.int32(left_lane_bias / left_lane_slope), 0
	left_lane = Line(x1, y1, x2, y2)

	right_lane_slope = np.median([line.slope for line in pos_lines])
	# print(right_lane_slope)
	right_lane_bias = np.median([line.bias for line in pos_lines])
	# print(right_lane_bias)
	x1, y1 = 0, right_lane_bias
	x2, y2 = image_width, np.int32(right_lane_slope*image_width + right_lane_bias)
	right_lane = Line(x1, y1, x2, y2)

	line_image = np.zeros((image_height, image_width))
	left_lane.draw(line_image)
	right_lane.draw(line_image)
	# cv2.imshow('line_image', line_image)
	# cv2.waitKey(0)
	line_image = region_of_interest(line_image, vertices)

	line_image = np.uint8(line_image)
	color_edges = np.dstack((line_image, np.zeros_like(line_image), np.zeros_like(line_image)))

	output = cv2.addWeighted(color_edges, 0.8, image, 1, 0)
	return output





if __name__ == '__main__':
	test_images = [os.path.join('test_images', name) for name in os.listdir('test_images')]
	for image in test_images:
		out_path = os.path.join('out', 'images', os.path.basename(image))
		input_image = cv2.cvtColor(cv2.imread(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		output_image = lane_detection_pipline(input_image)
		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_path, output_image)
		cv2.imshow('output', output_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()




















# test_images = [os.path.join('test_images', name) for name in os.listdir('test_images')]
# for test_image in test_images:
# 	img = cv2.imread(test_image, cv2.IMREAD_COLOR)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)
# 	img_edge = cv2.Canny(img_blur, threshold1 = 50, threshold2 = 80)
# 	# cv2.imshow('edges', img_edge)
# 	# cv2.waitKey(0)

# 	lines = cv2.HoughLinesP(img_edge, 2, np.pi/180, 1, 15, 7)

# 	print(lines.shape)
# 	print(type(lines))

# 	lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
# 	candinate_lines = []
# 	for line in lines:
# 		if 0.5 <= np.abs(line.slope) <= 2:
# 			candinate_lines.append(line)

# 	pos_lines = [l for l in candinate_lines if l.slope>0]
# 	print(len(pos_lines))
# 	neg_lines = [l for l in candinate_lines if l.slope<0]
# 	print(len(neg_lines))


# 	neg_bias = np.int32(np.median([l.bias for l in neg_lines]))
# 	print(neg_bias)
# 	neg_slope = np.median([l.slope for l in neg_lines])
# 	print(neg_slope)
# 	x1, y1 = 0, neg_bias
# 	x2, y2 =  -np.int32(neg_bias/neg_slope), 0
# 	left_lane = Line(x1, y1, x2, y2)



# 	pos_bias = np.int32(np.median([l.bias for l in pos_lines]))
# 	print(pos_bias)
# 	pos_slope = np.median([l.slope for l in pos_lines])
# 	print(pos_slope)
# 	x1, y1 = 0, pos_bias
# 	x2, y2 = img_gray.shape[1], np.int32(pos_slope*img_gray.shape[1] + pos_bias)
# 	# x2, y2 = np.int32(np.round((img_gray.shape[0] - pos_bias) / pos_slope)), img_gray.shape[0]
# 	print(x1, y1, x2, y2)
# 	right_lane = Line(x1, y1, x2, y2)


# 	line_img = np.zeros((img.shape[0], img.shape[1]))
# 	print(line_img.shape)
# 	left_lane.draw(line_img)
# 	right_lane.draw(line_img)
# 	# cv2.imshow('fr', line_img)
# 	# cv2.waitKey(0)


# 	mask = np.zeros_like(line_img)
# 	print(mask.shape)
# 	cv2.fillPoly(mask, np.array([[(450,320), (510,320), (910,540), (50,540)]]), 255)
# 	masked_img = cv2.bitwise_and(line_img, mask)
# 	# cv2.imshow('fr', mask)
# 	# cv2.waitKey(0)


# 	# img_line = np.copy(img)*0

# 	# for line in lines:
# 	#     for x1,y1,x2,y2 in line:
# 	#         cv2.line(img_line,(x1,y1),(x2,y2),(255,0,0),10)
# 	masked_img = np.uint8(masked_img)
# 	color_edges = np.dstack((masked_img, np.zeros_like(masked_img), np.zeros_like(masked_img)))

# 	lines_edges = cv2.addWeighted(color_edges, 0.8, img, 1, 0) 

# 	output = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB)

# 	cv2.imshow('output', output)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
