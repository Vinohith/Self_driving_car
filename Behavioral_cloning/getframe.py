import numpy as np 
import cv2
import mss
import time


with mss.mss() as sct:
	# moniter = {"top": 40, "left": 0, "width": 640, "height": 480}
	# while 'Screen capturing':
	# 	last_time = time.time()
	# 	img = np.array(sct.grab(moniter))
	# 	# cv2.imshow('frame', img)
	# 	print("fps: {}".format(1 / (time.time() - last_time)))
	# 	if cv2.waitKey(25) & 0xFF == ord("q"):
	# 		cv2.destroyAllWindows()
	# 		break 
	bbox = (50, 65, 50+1020, 65+350)
	img = np.array(sct.grab(bbox))
	cv2.imshow('frame', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	