from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage import transform, measure, feature
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st

def predict_image(img: np.ndarray):
    
    img = img[..., ::-1]
    img0 = cv2.imread('train/all.jpg')
    img1 = np.float32(img0[:,:,0])
    template = img1[132:183,2128:2182]
    img2 = np.float32(img[:,:,0])
    corr = feature.match_template(img2, template, pad_input=True)
    lbl, n = label(corr >= 0.5, connectivity=2, return_num=True)
    city_centers = np.int64([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])

    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    blue = (HUE > 90) & (HUE < 110) & (LIGHT < 80) & (SAT > 170)
    green =  (HUE > 50) & (HUE < 81)
    black = (SAT < 35) & (LIGHT < 30) 
    yellow = (HUE > 17) & (HUE < 30) & (SAT > 145)
    red = (HUE > 170) & (SAT > 130) &  (LIGHT > 70) & (LIGHT < 150)

    kernel = np.ones((15,15),np.uint8)
    kernel1= np.ones((20,20),np.uint8)

    def color_trains(color):
        x = cv2.morphologyEx(color.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        y = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel1)
        return y
    def find_if_close(cnt1,cnt2,d):
        row1,row2 = cnt1.shape[0],cnt2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i]-cnt2[j])
                if abs(dist) < d :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False
    def count_score(color):
        score = 0
        count = 0
        contours, hierarchy = cv2.findContours(color, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        L = len(contours)
        state = np.zeros((L,1))
        for i,cnt1 in enumerate(contours):
            x = i    
            if i != L-1:
                for j,cnt2 in enumerate(contours[i+1:]):
                    x = x+1
                    dist = find_if_close(cnt1,cnt2,30.5)
                    if dist == True:
                        val = min(state[i],state[x])
                        state[x] = state[i] = val
                    else:
                        if state[x]==state[i]:
                            state[x] = i+1
        unified = []
        maximum = int(state.max())+1
        for i in range(maximum):
            pos = np.where(state==i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                unified.append(hull)
        for contour in unified:
            perimeter = cv2.arcLength(contour, True) 
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour,epsilon,True)
            perimeter = cv2.arcLength(approx, True) 
            if perimeter > 270:
                if perimeter < 350:
                    count += 1
                    score += 1
                elif perimeter < 620:
                    count += 2
                    score += 2
                elif perimeter < 900:
                    count += 3
                    score += 4
                elif perimeter < 1400:
                    count += 4
                    score += 7
                elif perimeter < 1800:
                    count += 6
                    score += 15
                elif perimeter < 2000:
                    count += 8
                    score += 21
                elif perimeter > 2000:
                    count += 11
                    score += 18

        return count, score

    blue_trains = color_trains(blue)
    green_trains = color_trains(green)
    black_trains = color_trains(black)
    yellow_trains = color_trains(yellow)
    red_trains = color_trains(red)

    blue_count, blue_score = count_score(blue_trains)
    green_count, green_score = count_score(green_trains)
    black_count, black_score = count_score(black_trains)
    yellow_count, yellow_score = count_score(yellow_trains)
    red_count, red_score = count_score(red_trains)

    n_trains = {'blue': blue_count, 'green': green_count, 'black': black_count, 'yellow': yellow_count, 'red': red_count}
    scores = {'blue': blue_score, 'green': green_score, 'black': black_score, 'yellow': yellow_score, 'red': red_score}
    return city_centers, n_trains, scores