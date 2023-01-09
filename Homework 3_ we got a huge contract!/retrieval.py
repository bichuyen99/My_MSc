import cv2
import numpy as np

def predict_image(img, query):
    img1 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY) # queryImage
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # trainImage

    image_height, image_width = img2.shape[:2]
    
    MIN_MATCH_COUNT = 10
    list_of_bboxes = []
    def SIFT(img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        FLANN_INDEX_KDTREE = 5
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        search_params = dict(checks = 500)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance <0.95*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            transformedCorners = cv2.perspectiveTransform(corners,M)
            img2 = cv2.polylines(img2,[np.int32(transformedCorners)],True,255,10, cv2.LINE_AA)
            x_min = (np.int32(transformedCorners)[0][0][0])/image_width
            y_min = (np.int32(transformedCorners)[0][0][1])/image_height
            width = img1.shape[1]/image_width
            height = img1.shape[0]/image_height
            data = [x_min, y_min, width, height]
            list_of_bboxes.append(tuple(data))
            img2 = cv2.fillPoly(img2,[np.int32(transformedCorners)], color=(0,0,0))
            S1 = cv2.contourArea(np.int32(transformedCorners))
        else:
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        return S1
    
    S1 = SIFT(img1, img2)
    S2 = S1
    if S2 != 0:
      while (2 >= S1/S2 >= 0.5):
          S1 = S2
          S2 = SIFT(img1, img2)
    return(list_of_bboxes)
