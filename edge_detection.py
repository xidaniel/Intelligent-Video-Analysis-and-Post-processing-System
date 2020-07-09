import cv2
import numpy as np
import random
import colorsys
import enhance


def findPoints(contours):
    """
    :param contours: numpy datatype
            contours = [array([[[230, 240]],

                               ...,

                               [[231, 240]]], dtype=int32)]
    :return: list of rect
    """

    approx = contours[0].reshape(len(contours[0]), 2)

    rect = np.zeros((4, 2), np.int32)
    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)]  # top-left
    rect[3] = approx[np.argmax(s)]  # bottom-right

    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)]  # top-right    diff = a[n+1] - a[n]
    rect[2] = approx[np.argmax(diff)]  # bottom-left

    return rect.tolist()


def fourPointsTransform(img, rect):
    """
    :param img: numpy
    :param rect: list
    :return:
    """

    rect = np.float32(rect)
    (tl, tr, bl, br) = rect

    widthA = br[0] - bl[0]
    widthB = tr[0] - tl[0]
    maxWidth = max(widthA, widthB)

    heightA = bl[1] - tl[1]
    heightB = br[1] - tr[1]
    maxHeight = max(heightA, heightB)

    dst = np.float32([[0, 0],
                      [maxWidth - 1, 0],
                      [0, maxHeight - 1],
                      [maxWidth - 1, maxHeight - 1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


def getArea(rect):
    '''
    :param rect: list within list with 4 length
    :return: int area
    '''
    coord = np.array(rect).reshape((4, 2))

    # vector point(x2-x1, y2-y1)
    vectorSet = np.empty([4, 2], dtype=int)
    vectorSet[0] = coord[1] - coord[0]
    vectorSet[1] = coord[2] - coord[0]
    vectorSet[2] = coord[1] - coord[3]
    vectorSet[3] = coord[2] - coord[3]

    # compute the determinant
    area1 = abs(np.linalg.det(np.array([vectorSet[0], vectorSet[1]]))) / 2
    area2 = abs(np.linalg.det(np.array([vectorSet[2], vectorSet[3]]))) / 2

    return int(area1 + area2)


def nextPoint(contours, point1, point2):
    """
    To estimate where is next point based on two fixed points
    :param contours: filtered contours list  [[[]], [[]], [[]]]
    :param point1: list []
    :param point2: list []
    :return: list []
    """
    cnt = contours[0].tolist()
    # y = ax + b
    secondePoint1 = cnt[cnt.index([point1]) - len(cnt) // 18][0]  # bottom left
    x1, y1 = point1[0], point1[1]
    z1, k1 = secondePoint1[0], secondePoint1[1]
    a1, b1 = slope(x1, y1, z1, k1)

    secondePoint2 = cnt[cnt.index([point2]) + len(cnt) // 10][0]  # top right
    x1, y1 = point2[0], point2[1]
    z1, k1 = secondePoint2[0], secondePoint2[1]
    a2, b2 = slope(x1, y1, z1, k1)

    x = int((b2 - b1) / (a1 - a2))
    y = int(((a2 * b1) - (a1 * b2)) / (a2 - a1))

    return [x, y]


def slope(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = ((x2 * y1) - (x1 * y2)) / (x2 - x1)
    return a, b


def top_K(contours, k):
    """
    input: contours
    Contours is a Python list of all the contours in the image.
    Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.

    contours = list[numpy([[1,2]]),]
    contours = [
                ([[[1,2]],[[3,4]]]),
                ([[[4,5]],[[7,4]]])
                ]

    k: numbers of return value
    :return: index of contours
    """
    sets = []
    for i, contour in enumerate(contours):
        point = len(contour)
        fourPoints = findPoints([contour])
        area = getArea(fourPoints)
        sets.append((area, point, i))

    tops = sorted(sets, key=lambda x: (x[0], x[1]), reverse=True)[:k]

    result = []
    for ele in tops:
        result.append(contours[ele[2]])

    return result


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)
    edged = cv2.Canny(image, lower, upper)

    return edged


def detectContours(img, lower=40, upper=200):
    """
    :param img: original img numpy
    :param lower: lower threshold
    :param upper: upper threshold
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = (5, 5)
    blur = cv2.GaussianBlur(gray, kernel, 0)
    edged = cv2.Canny(blur, lower, upper, True)
    img2, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    colors = [tuple(round(j * 255) for j in colorsys.hsv_to_rgb(i / n, 1, brightness)) for i in range(n)]
    random.shuffle(colors)
    return colors


def drawArea(contours, canvas):
    """
    :param contours: numpy
    :param canvas: black background, size same as original img
    :return: None
    """
    n = len(contours)
    colors = random_colors(n)
    for i in range(len(contours)):
        b, g, r = colors[i]
        cv2.drawContours(canvas, contours[i], -1, (b, g, r), 2)
        fourPoints = findPoints([contours[i]])
        area = getArea(fourPoints)

        text = 'Area: {}'.format(area)
        cv2.putText(canvas, text, (fourPoints[0][0] + 5, fourPoints[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (b, g, r), 1, cv2.LINE_AA)


def drawFourPoints(contours, canvas):
    """
    :param contours: numpy
    :param canvas:
    :return:
    """
    n = len(contours)
    colors = random_colors(n)
    for i in range(len(contours)):
        rect = findPoints([contours[i]])
        b, g, r = colors[i]
        for j in range(len(rect)):
            cv2.circle(canvas, tuple(rect[j]), 2, (b, g, r), 5)
            cv2.putText(canvas, str(j), tuple(rect[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)


def test():
    img = cv2.imread('images/111.jpg')
    # img = enhance.brightness(img,100)
    image = img.copy()
    canvas = np.zeros(img.shape, np.uint8)
    canvas1 = np.zeros(img.shape, np.uint8)
    # edged = auto_canny(blur)

    # contoursDetection
    contours = detectContours(img)

    # draw all detected contours
    drawArea(contours, canvas)
    # drawFourPoints(contours, canvas)
    text = str(len(contours)) + " contour(s) be found"
    cv2.putText(canvas, text, (0, canvas.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # contours filtering
    filtered = top_K(contours, 1)

    # draw filtered contour
    drawArea(filtered, canvas1)
    # drawFourPoints(filtered, canvas1)
    # drawFourPoints(filtered, image)

    # find four points and update
    rect = findPoints(filtered)
    print("top-left:{}, top-right:{},  bottom-left:{}, bottom-right:{}".format(rect[0], rect[1], rect[2], rect[3]))

    fourthPoint = nextPoint(filtered, rect[1], rect[2])
    rect[3] = fourthPoint
    for ele in rect:
        cv2.circle(image, tuple(ele), 2, (0, 0, 255), 5)
        cv2.circle(canvas1, tuple(ele), 2, (0, 0, 255), 5)
    print(rect[3])

    # correct image
    # rect = findPoints(filtered)
    warped = fourPointsTransform(img, rect)
    warped_buffing = enhance.buffing(warped)

    # image enhance
    # warped1 = enhance.buffing(warped)

    # show image
    images = [img, canvas, canvas1, image, warped, warped_buffing]
    for img in images:
        cv2.imshow('img', img)
        cv2.waitKey()


test()
