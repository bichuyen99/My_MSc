from skimage.feature import canny
import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import rescale, resize, ProjectiveTransform, warp
import operator
import warnings
warnings.filterwarnings('ignore')

"""Part 1: Find tables \\
Step 1: find some keypoints (Otsu thresholding, edges, Hough lines, corners of 9x9 table, etc.) \\
Step 2: find 9x9 tables, estimate the Sudoku-ness of every table (Hough lines, regular structures, etc.) \\
Step 3: apply Projective Transform for every found table
"""
def Threshold(img):
    img = cv2.GaussianBlur(img,(9,9),0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
    div = np.float32(gray)/(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel))
    res = np.uint8(cv2.normalize(div,div,1,255,cv2.NORM_MINMAX))
    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    return thresh, res, gray 

def find_contour(img):
    contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    cnt = []
    for x in contours: 
      if cv2.contourArea(x) > 1000:
        cnt.append(x)
    return cnt

def mask_img(res, cnt):   
    mask = np.zeros((res.shape), np.uint8)
    points = []
    for i in range(len(cnt) - 1):
        if i == 0:
            cv2.fillConvexPoly(mask, cnt[0], 255)
            count = 1
        elif (cv2.contourArea(cnt[i+1]) / cv2.contourArea(cnt[i]))< 0.05:
            cv2.fillConvexPoly(mask, cnt[0], 255)
            count += 1
    res = cv2.bitwise_and(res, mask)
    return mask, count

def find_corners(poly):
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in poly]), key=operator.itemgetter(1))
    return [poly[top_left][0], poly[top_right][0], poly[bottom_right][0], poly[bottom_left][0]]

def dist(d1, d2):
  return np.sqrt((d2[0] - d1[0]) ** 2 + (d2[1] - d1[1]) ** 2)

def split_boxes(board):
    input_size = 28
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes

def matrix(rowCount, colCount, dataList):
    mat = []
    for i in range(rowCount):
        rowList = []
        for j in range(colCount):
            rowList.append(dataList[rowCount * i + j])
        mat.append(rowList)
    return mat

"""Part 2: Recognize digits \\

Step 4: divide the table into separate cells (optionally: remove table artifacts) \\
Step 5: build digit classifier on MNIST or manually (semi-supervised) annotated train data: feature extractor (e.g. HoG) + classifier (SVM, Random Forest, NN, etc.)
"""
def sudoku_board(img, count, contours):
    model = tf.keras.models.load_model('/autograder/submission/model.h5')
    polygon = contours[count]
    corners = find_corners(polygon)
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    smax = max([
            dist(bottom_right, top_right),
            dist(top_left, bottom_left),
            dist(bottom_right, bottom_left),
            dist(top_left, top_right)
          ])
    dst = np.array([[0, 0], [smax - 1, 0], [smax - 1, smax - 1], [0, smax - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(img, m, (int(smax), int(smax)))
    side = max(crop.shape)
    crop = resize(crop, (side, side), anti_aliasing=True)
    crop = rescale(crop, 900/side)
    img_norm = (crop - crop.mean())/crop.std()
    result = np.uint8(img_norm).copy()
    thresh = cv2.threshold(np.uint8(img_norm), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    squares = split_boxes(thresh)
    rois = []
    for i in range(len(split_boxes(thresh))):
        for k in range(27):
            for j in range(5):
                squares[i][k,j] = 1
                squares[i][j,k] = 1
                squares[i][k,27-j] = 1
                squares[i][27,27] = 1
                squares[i][j-5,k] = 1
        rois.append(1 - squares[i])    
    rois = np.array(rois)
    rois = rois.reshape(-1, 28, 28, 1)
    classes = np.arange(1, 10)
    prediction = model.predict(rois)
    predicted_numbers = []
    ind = 0
    for i in range(81):
        if prediction[i].max() > 0.5:
            index = np.argmax(prediction[i])
            predicted_number = classes[index]
            predicted_numbers.append(predicted_number)
            ind += 1
        else:
            predicted_number = -1
            predicted_numbers.append(predicted_number)
            ind +=1
    digits = np.array(matrix(9, 9, predicted_numbers))
    return digits

  
def predict_image(image: np.array) -> (np.array, list):
    thresh, res, gray = Threshold(image)
    contours = find_contour(thresh)
    mask, count = mask_img(res, contours)
    list_arrays = []
    for i in range(count):
        digits = sudoku_board(gray, i, contours)
        list_arrays.append(digits)
    return mask, (list_arrays)

