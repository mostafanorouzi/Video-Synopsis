import numpy as np
from scipy.stats import itemfreq
import cv2


# classes
class tube:
    def __init__(self, cx, cy, x, y, w, h, t_sec, img):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t_sec = round(t_sec, 2)
        self.target = np.array(img[y:y + w, x:x + h, :])


class MyCar:
    def __init__(self, i, tube: object, frameId):
        self.i = i
        self.x = tube.x
        self.y = tube.y
        self.cx = tube.cx
        self.cy = tube.cy
        self.w = tube.w
        self.h = tube.h
        self.tubes = []
        self.tracks = []
        self.R = tube.r
        self.G = tube.g
        self.B = tube.b
        self.done = False
        self.state = '0'
        self.dir = None
        self.tubes.append(tube)
        self.empty = False
        self.start = 0
        self.endFrame = 1
        self.frameId = frameId
        self.speed = 0
        self.startFrame = 0

    def getRGB(self):
        return (self.R, self.G, self.B)

    def begin(self):
        return self.tubes[self.start]

    def end(self):
        return self.tubes[0]

    def empty(self):
        return self.empty

    def lentube(self):
        if self.empty:
            return 0
        return len(self.tubes) - self.start

    def pop_front(self):
        if self.start == (self.lentube() - 1):
            self.empty = True
        else:
            self.start += 1
            self.startFrame += 1

    def getId(self):
        return self.i

    def getTracks(self):
        return self.tracks

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getCX(self):
        return self.cx

    def getCY(self):
        return self.cy

    def setDone(self):
        self.done = True

    def updateCoords(self, tube: object, frameId):
        self.tracks.append([tube.cx, tube.cy])
        self.x = tube.x
        self.y = tube.y
        self.w = tube.w
        self.h = tube.h
        self.cx = tube.cx
        self.cy = tube.cy
        self.tubes.append(tube)
        self.frameId = frameId
        self.R = tube.r
        self.G = tube.g
        self.B = tube.b
        self.endFrame += 1

    def going_UP(self, line_down, line_up):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if self.tracks[-1][1] < line_up <= self.tracks[0][1]:
                    self.state = '1'
                    self.dir = 'up'
                    return True
            else:
                return False
        else:
            return False

    def going_DOWN(self, line_down, line_up):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if self.tracks[-1][1] > line_down >= self.tracks[0][1]:
                    self.state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False

    def getSpeed(self):
        if len(self.tracks) >= 3:
            return (self.tracks[-1][1] - self.tracks[-3][1]) / 3
        else:
            return 0


# functions
def isOverLabRects(a, b):
    if (a[0] > b[0]):
        x = a
        a = b
        b = x
    l = a
    r = b
    if (l[2] < r[0] - l[0]):
        return False
    elif (l[1] < r[1] + 1 and l[3] >= r[2] - l[2]):
        return True
    elif (l[1] > r[1] and r[3] >= l[1] - r[1]):
        return True
    else:
        return False


def isOverLabCar(car1, car2):
    if car2.startFrame > car1.lentube():
        return False

    for j in range(max(car2.startFrame, car1.startFrame), min(car1.endFrame, car2.endFrame)):
        a = (car2.tubes[j].x, car2.tubes[j].y, car2.tubes[j].w, car2.tubes[j].h)
        b = (car1.tubes[j].x, car1.tubes[j].y, car1.tubes[j].w, car1.tubes[j].h)
        if isOverLabRects(a, b):
            return True
    return False


def getBG(path, randomFrameSize):
    # Open Video
    cap = cv2.VideoCapture(path)

    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=randomFrameSize)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # Calculate the median along the time axis
    img_b = np.median(frames, axis=0).astype(dtype=np.uint8)
    return img_b


def gradientline(frame, pts):
    ans = pts.shape[0]
    for i in range(ans - 1):
        cv2.polylines(frame, [pts[i:i + 2]], False, (int(255 / ans * i), int(255 / ans * i), int(255 / ans * i)), 4)
    return np.array(frame)


def setColor(tube):
    slice_bg = np.array(tube.target)
    arr = np.float32(slice_bg)
    # reshaping the image to a linear form with 3-channels
    pixels = arr.reshape((-1, 3))

    # number of clusters
    n_colors = 2

    # number of iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .1)

    # initialising centroid
    flags = cv2.KMEANS_RANDOM_CENTERS

    # applying k-means to detect prominant color in the image
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]

    # detecting the centroid with densest cluster
    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    tube.r = int(dominant_color[0])
    tube.g = int(dominant_color[1])
    tube.b = int(dominant_color[2])


def keyPointSimilarity(a, b):
    criteria = 0
    if a.shape[0] > b.shape[0]:
        gray1 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.AKAZE_create()
    (kps1, features1) = descriptor.detectAndCompute(gray1, None)
    (kps2, features2) = descriptor.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if len(kps2) and len(kps1):
        matches = bf.knnMatch(features1, features2, k=2)
        good = []
        list_kps1 = []
        list_kps2 = []
        if len(matches) > 0 and len(matches[0]) > 1:
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * .8:
                    good.append([m])
                    img1_idx = m[0].queryIdx
                    img2_idx = m[0].trainIdx
                    # print(len(kps1),img1_idx)
                    if img2_idx >= len(kps2):
                        print(len(kps2), img2_idx)
                    (x1, y1) = kps1[img1_idx].pt
                    # (x2, y2) = kps1[img2_idx].pt
                    # print(img2_idx)
                    list_kps1.append((x1, y1))
                    # list_kps2.append((x2, y2))
            pts1 = np.array(list_kps1)
            print(pts1.shape)
            # print(len(kps1))
            if pts1.shape[0] > 0:
                criteria = pts1.shape[0] / len(kps1)

    return criteria
