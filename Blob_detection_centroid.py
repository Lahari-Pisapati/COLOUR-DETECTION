import cv2
import numpy as np

#detecting red
img = cv2.imread('30-legos.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([100,100,100])
upper_red = np.array([180,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
ret, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY_INV)
mask = cv2.erode(mask, None, iterations = 5)
mask = cv2.dilate(mask, None, iterations = 1)
#print(ret)
# find contours in the binary image
im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
   # calculate moments for each contour
   M = cv2.moments(c)
 
   # calculate x,y coordinate of center
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
   #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(mask)
print("red cubes")
print(len(keypoints))

# Draw detected keypoints as red circles
imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#detecting green
lower_green = np.array([30,100,100])
upper_green = np.array([90,255,255])

mask1 = cv2.inRange(hsv, lower_green, upper_green)
ret1, mask1 = cv2.threshold(mask1, 120, 255,cv2.THRESH_BINARY_INV)
mask1 = cv2.erode(mask1, None, iterations = 5)
mask1 = cv2.dilate(mask1, None, iterations = 1)
im3, contours1, hierarchy1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours1:
   # calculate moments for each contour
   M = cv2.moments(c)
 
   # calculate x,y coordinate of center
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   cv2.circle(imgKeyPoints, (cX, cY), 5, (255, 255, 255), -1)
   #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.imshow("green", mask1)
keypoints1=detector.detect(mask1)
print("green cubes")
print(len(keypoints1))
imgKeyPoints = cv2.drawKeypoints(imgKeyPoints, keypoints1, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("image", img)
cv2.imshow("mask", mask)
cv2.imshow("keypoints",imgKeyPoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
