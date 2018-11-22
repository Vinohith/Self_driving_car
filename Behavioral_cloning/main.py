import numpy as np
import mss
import mss.tools
from PIL import Image
import cv2



with mss.mss() as sct:
	mon = {'top': 10, 'left': 10, 'width': 640, 'height': 480}
	sct_img = sct.grab(mon)
	# img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
	# mss.tools.to_png(sct_img.rgb, sct_img.size)
	img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
	# cv2.imshow('frame', np.array(img))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()