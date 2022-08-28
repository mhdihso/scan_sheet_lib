import cv2


import operator


import cv2
import numpy as np
import imutils


class Shape:
    def __init__(self, _x, _y, _w, _h):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h

    def getTotalW(self):
        return self.x + self.w

    def getTotalY(self):
        return self.y + self.h

    def __lt__(self, other):
        return self.x < other.x


def color_seg():
    lower_hue = np.array([0, 0, 0])
    upper_hue = np.array([50, 50, 100])
    return lower_hue, upper_hue


def getPositionedColor(frame):
    # Take each frame


    # frame = cv2.imread('images/road_1.jpg')

    chosen_color = 'black'
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of a color in HSV
    lower_hue, upper_hue = color_seg()
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=10)
    erosion = cv2.filter2D(mask, -1, kernel)
    erosion = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
    i = 0
    lists = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        nesbat = w / h

        i = i + 1
        lists.append(Shape(x, y, w, h))


        # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.putText(mask, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.23, 255)

    return lists


def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def getSquare(image):


    src = image

    # HSV thresholding to get rid of as much background as possible
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))

    # Adaptive Thresholding to isolate the bed
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    canvas = src.copy()
    hei = (src.shape[1]) - 5
    wid = (src.shape[1]) - 5

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 25000 and w * h <(hei*wid):
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)
    x, y, w, h = brightest_rectangle
    # cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.imshow("crop", canvas)
    # cv2.waitKey(0)
    return brightest_rectangle


def getAnswers(startfrom, mylist, first , answersHelpers):
    mylist.sort()


    returnlist = []
    for shape in mylist:
        trueDot = shape.x + shape.w / 2
        for ii in range(24):
            varr = 23 - ii
            answers = answersHelpers[varr]

            if answers.x - startfrom < trueDot < answers.getTotalW() - startfrom:
                returnlist.append(varr + 1)
                break
    dic = {}
    for rl in returnlist:
        mh = rl // 4
        bg = rl % 4
        if bg == 0:
            bg = 4
            mh = mh - 1
        if dic.get((mh * 50) + first, None) is None:
            dic[(mh * 50) + first] = set()
        dic[(mh * 50) + first].add(bg)

    return dic

import numpy as np
import imutils
import cv2
import skimage
from skimage.filters import threshold_local
import matplotlib.pyplot as plt


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # the top-right point will have the smallest difference, whereas the bottom-left will have the largest
    # difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destinationPoints = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # compute the perspective transform matrix and then apply it
    mapped = cv2.getPerspectiveTransform(rect, destinationPoints)
    warped = cv2.warpPerspective(image, mapped, (maxWidth, maxHeight))

    return warped


def getPapar(src):
    plt.rcParams['figure.figsize'] = [16, 10]

    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it

    image = cv2.imread(src)
    ratio = image.shape[0] / 1000.0
    orig = image.copy()
    image = imutils.resize(image, height=1000)
    # convert the image to grayscale, blur it, and find edges
    # in the image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gary = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)  # minVal, maxVal

    dispImage = cv2.cvtColor(edged.copy(), cv2.COLOR_GRAY2RGB)
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    flag = True
    for c in contours:

        perimeter = cv2.arcLength(c, True) 
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx) == 3:

            cv2.drawContours(image, [c], 0, (0, 255, 0), 3)

            cv2.imwrite('fffff.png', image)

            cv2.waitKey(0)

            print("founded")
            break

        if len(approx) == 4:
            #
            flag = False


            for c1 in approx:

                for c2 in c1:
                    for c3 in c2:
                        continue
            screenContour = approx
            break

    if (flag):
        width = (image.shape[1])
        x = [[width, 0], [0, 0], [0, 1000], [width, 1000]]
        aa_3d_array = np.array(x)

        screenContour = aa_3d_array
    cv2.drawContours(image, [screenContour], -1, (255, 0, 0), 2)  # -1 => draw all contours, (color), thickness
    dispImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenContour.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    dispImage = imutils.resize(warped, height=1000)
    cp_im = f"cp_{src}"
    cv2.imwrite(cp_im, dispImage)
    return cp_im


class Scan:
    def __init__(self, image):
        self.image = image

    def scan(self):
        answersHelpers = []
        allAnswers = {}
        src = str(self)
        image = cv2.imread(getPapar(src))
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)

        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        rights = {}
        lefts = {}
        ie = 0
        io = 0
        i = 0
        x, y, w, h = getSquare(image)
        limitX1 = x + 1
        limitX2 = x + w - 1
        for c in cnts:

            x, y, w, h = cv2.boundingRect(c)
            sh = Shape(x, y, w, h)
            nesbat = w / h

            if (nesbat > 1.5 and nesbat < 4 and h > 4 and h < 12 and ((x > limitX2))):

                rights[ie] = sh
                ie = ie + 1

            ##left
            elif (nesbat > 1.5 and nesbat < 4 and h > 4 and h < 11 and ((x < limitX1 and x > 0))):

                lefts[io] = sh

                io = io + 1

        limitHorY1 = rights[49].y - rights[49].h - 1
        limitHorY2 = y + 1
        j = 0
        for c2 in cnts:
            x, y, w, h = cv2.boundingRect(c2)
            sh = Shape(x, y, w, h)
            nesbat = w / h

            if limitHorY2 < y < limitHorY1 and h < 100:
                answersHelpers.append(sh)
                j = j + 1

        answersHelpers.sort()
        for mm in range(0, 50):
            var = 49 - mm
            cv2.rectangle(image, (rights[var].x, rights[var].y), (rights[var].getTotalW(), rights[var].getTotalY()),
                        (36, 255, 12),
                        2)
            cv2.rectangle(image, (lefts[var].x, lefts[var].y), (lefts[var].getTotalW(), lefts[var].getTotalY()),
                        (36, 255, 12),
                        2)
            start = lefts[var].y - 4
            crop_img = image[start: rights[var].getTotalY() + 4, lefts[var].x + lefts[var].w: rights[var].x]

            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            canny = cv2.Canny(blurred, 120, 255, 1)
            choices = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            choices = choices[0] if len(choices) == 2 else choices[1]

            cppp = crop_img
            lists = getPositionedColor(cppp)

            shapes = []
            for c in lists:
                nesbat = c.w / c.h
                if (c.h >= 2 and c.w >= 2):
                    shapes.append(c)
                    cv2.rectangle(crop_img, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 0, 0), 2)
            dics = getAnswers(lefts[var].x + lefts[var].w, shapes, mm + 1 , answersHelpers)
            allAnswers.update(dics)
            allAnswers = dict(sorted(allAnswers.items(), key=lambda x: x[0], reverse=False))

        det=cv2.QRCodeDetector()
        val, pts, st_code=det.detectAndDecode(image)
        allAnswers['qr_code'] = val
        return allAnswers
