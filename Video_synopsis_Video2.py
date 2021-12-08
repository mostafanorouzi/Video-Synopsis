import cv2
import numpy as np
import copy
import Car
from Car import isOverLabCar, getBG, gradientline, setColor , isOverLabRects

video_path = 'Video2.avi'

# Open video file
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
capture_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('result_Video2.avi', fourcc, 20, capture_size)

# Create the background substractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Structuring elements for morphographic filters
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

# Variables
cars = []
font = cv2.FONT_HERSHEY_SIMPLEX
pid = 1
fps = cap.get(cv2.CAP_PROP_FPS)
frameId = 0
w = cap.get(3)
h = cap.get(4)
frameArea = h*w
areaTH = frameArea/600
up_limit = int(.4*(h/5))
down_limit = int(3*(h/5))

# create background
img_b = getBG(video_path, 40)

while (cap.isOpened()):
    # read a frame
    ret, orginalframe = cap.read()
    frame = copy.copy(orginalframe)

    # time : second
    t_sec = frameId / fps

    # Use the substractor
    fgmask = fgbg.apply(frame)

    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # Opening (erode->dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        # Closing (dilate -> erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        #additional image processing

    except:
        # if there are no more frames to show...
        print('EOF')
        break

    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            #################
            #   TRACKING    #
            #################
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            tube = Car.tube(cx,cy,x, y, w, h, t_sec, orginalframe)
            #calcute color of car
            setColor(tube)
            new = True
            if cy in range(up_limit, down_limit):
                for i in cars:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h and abs(i.frameId - frameId) < 5:
                        # the object is close to one that was already detected before
                        new = False
                        i.updateCoords(tube, frameId)
                        break
            if new == True:
                p = Car.MyCar(pid, tube, frameId)
                cars.append(p)
                pid += 1

    frameId += 1
#finish loop


for i in range(1,len(cars)) :
    for j in range(i) :
        while isOverLabCar(cars[i],cars[j]) :
            cars[i].startFrame+=1


#create video synopsis
n=0
while True:
    finish = True
    for i in cars :
        if i.lentube()>20:
            finish = False
    if finish :
        break
    frame = np.array(img_b)
    rects = []
    for i in range(len(cars)):
        if cars[i].lentube()>10 and cars[i].startFrame>=n  :
            temp = cars[i].begin()
            canAdd = True
            new_rect = (temp.x,temp.y,temp.w,temp.h)
            for rect in rects :
                if isOverLabRects(rect, new_rect) :
                    canAdd = False
            if canAdd :
                rects.append(new_rect)
                frame[temp.y:temp.y + temp.w, temp.x:temp.x + temp.h] = temp.target
                cv2.putText(frame, str(temp.t_sec), (temp.x, temp.y), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
                cars[i].pop_front()
    out.write(frame)
    cv2.imshow('result', frame)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cap.release()  # release video file
out.release()
cv2.destroyAllWindows()  # close all openCV windows
