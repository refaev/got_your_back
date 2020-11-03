import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import cv2 as cv
import os


def read_pkl(file_path):
    obj = pd.read_pickle(file_path)
    return obj

def show_results(obj , folder):
    for im_num in obj.keys():
        data = obj[im_num]
        file = data['fileName'][12:]
        file_path = os.path.join(folder, file)
        im = cv.imread(file_path)#[:,:,::-1]
        detections = data['detections']
        for detection in detections:
            type = detection[0]
            roi = detection[2]
            roi_tl = [roi[0] - roi[2]/2, roi[1] - roi[3]/2, roi[2], roi[3]]
            roi_ = np.asarray(roi_tl).astype(int)

            im = cv.rectangle(img=im, rec=roi_,  color=(255, 0, 0), thickness=3)

            plt.imshow(im[:, :, ::-1])
            plt.pause(0.0001)
            a=1

def track_objects(obj , folder):
    for im_num in obj.keys():
        data = obj[im_num]
        file = data['fileName'][12:]
        file_path = os.path.join(folder, file)
        im = cv.imread(file_path)#[:,:,::-1]
        detections = data['detections']
        for detection in detections:
            # type = detection[0]
            roi = detection[2]

            cen = [roi[0], roi[1]]
            if cen[1] > 500 and cen[0] > 500 and cen[1] < 1300:
                roi_tl = [roi[0] - roi[2] / 2, roi[1] - roi[3] / 2, roi[2], roi[3]]
                roi_ = np.asarray(roi_tl).astype(int)
                im = cv.rectangle(img=im, rec=roi_,  color=(0, 0, 255), thickness=3)
                continue

        out_path = os.path.join(folder, 'warning', file)
        cv.imwrite(out_path, im)
        a=1

if __name__ == '__main__':
    folder = r'E:\rafi\got_your_back\data\results_files\res\temp_dir - Copy (9)'
    file_path = os.path.join(folder, r"YoloV3_res\res_pkl.pkl")
    obj = read_pkl(file_path)
    track_objects(obj, folder)
    # show_results(obj, folder)