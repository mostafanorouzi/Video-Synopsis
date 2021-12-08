import numpy as np
import cv2
import Car
import time
from Car import gradientline, setColor

# Input and output counters
cnt_up   = 0
cnt_down = 0

#Video source
cap = cv2.VideoCapture('Video1.avi')

#Print the capture properties to console
for i in range(19):
    print (i, cap.get(i))

w = cap.get(3)
h = cap.get(4)
frameArea = h*w
frameId = 0
fps = cap.get(cv2.CAP_PROP_FPS)
areaTH = frameArea/850
print ('Area Threshold', areaTH)

#In / out lines
line_up = int(2.4*(h/5))
line_down   = int(2.7*(h/5))

up_limit = int(1.6*(h/5))
down_limit = int(3*(h/5))

print ("Red line y:",str(line_down))
print ("Blue line y:", str(line_up))

line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
capture_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('result_video1_tracking.avi', fourcc, 20, capture_size)

# Structuring elements for morphographic filters
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Background Substractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
pid = 1

while(cap.isOpened()):
    #Read an image of the video source
    ret, orginalframe = cap.read()  # read a frame

    frame = np.array(orginalframe)

    # time : second
    t_sec = frameId / fps

    #Application background subtraction
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    #Binariazcion to eliminate shadows (gray color)
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #Opening (erode->dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp2)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp2)
        #Closing (dilate -> erode)
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print ('UP:',cnt_up)
        print ('DOWN:',cnt_down)
        break

    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            #################
            #   TRACKING    #
            #################
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue
            tube = Car.tube(cx,cy,x, y, w, h, t_sec, orginalframe)

            #calcute color of car
            setColor(tube)

            new = True
            if cy in range(up_limit,down_limit):
                for i in cars:
                    if abs(cx-i.getCX()) <= w and abs(cy-i.getCY()) <= h and abs(i.frameId - frameId) <10 :

                        # the object is close to one that was already detected before
                        new = False
                        i.updateCoords(tube,frameId)

                        RGB = i.getRGB()

                        if i.going_UP(line_down,line_up) == True and i.done==0 :
                            cnt_up += 1;
                            i.setDone()
                            print ("ID:",i.getId(),'crossed going up at',time.strftime("%c"))
                        elif i.going_DOWN(line_down,line_up) == True and i.done==0:
                            cnt_down += 1;
                            i.setDone()
                            print ("ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                        break

                if new == True:
                    p = Car.MyCar(pid, tube, frameId)
                    RGB =p.getRGB()
                    cars.append(p)
                    pid += 1

                cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),RGB,2)
    #END for cnt in contours0

    #Draw gradiant line
    for i in cars:
        if len(i.getTracks()) >= 2:
           pts = np.array(i.getTracks(), np.int32)
           pts = pts.reshape((-1,1,2))
           frame = gradientline(frame,pts)

    #################
    #   IMAGES      #
    #################
    #frame = orginalframe
    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    #frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    #frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.imshow('Frame',frame)

    out.write(frame)
    #press ESC to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    frameId+=1
#END while(cap.isOpened())


cap.release()
out.release()
cv2.destroyAllWindows()
