import cv2
import numpy as np

def orderPoints(image):
    """
    :param screen_bbox: the screen_bbox[x,y,w,h]
    :return:
    """
    #img = frame[screen_bbox_y:screen_bbox_y + screen_bbox_h, screen_bbox_x:screen_bbox_x + screen_bbox_w]
    colorSpace = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    blur = cv2.GaussianBlur(colorSpace, (5, 5), 0)
    edged = cv2.Canny(blur, 40, 255)

    image, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    con_size = [len(contours[i]) for i in range(len(contours))]
    index = con_size.index(max(con_size))
    #cv2.drawContours(img, contours, index, (0, 255, 0), 3)
    approx = contours[index].reshape(len(contours[index]), 2)

    rect = np.zeros((4,2),np.float32)
    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)] # top-left
    rect[2] = approx[np.argmax(s)] # bottom-right

    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)] # top-right    diff = a[n+1] - a[n]
    rect[3] = approx[np.argmax(diff)] # bottom-left
    return rect

def fourPointsTransform(img,rect):
    (tl,tr,br,bl) = rect

    widthA = br[0] - bl[0]
    widthB = tr[0] - tl[0]
    maxWidth = max(widthA,widthB)

    heightA = bl[1] - tl[1]
    heightB = br[1] - tr[1]
    maxheight = max(heightA,heightB)

    dst = np.float32([[0,0],[maxWidth,0],[maxWidth,maxheight],[0,maxheight]])
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(img, M, (maxWidth,maxheight))
    return  warped

# Main code

# img = cv2.imread(img_list[2])
#
# rect = orderPoints(img)  # find and order four points
#
# warped = fourPointsTransform(img,rect) # perspective transform

### method 2  ####
#area = cv2.contourArea(contours[index])
#epsilon  = 0.1*cv2.arcLength(contours[index],True)   # Douglas_Peucker Algorithm
#approx = cv2.approxPolyDP(contours[index],epsilon,True)
#approx = approx.reshape(4,2)
#print(approx)

# drawing points
# points_list = [tuple(ele) for ele in rect.astype(np.int16).tolist()]
# for ele in points_list:
#     cv2.circle(img,ele,2,(0,0,255),2)
#
# cv2.imshow('Original',img)
# cv2.imshow('Warped',warped)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()