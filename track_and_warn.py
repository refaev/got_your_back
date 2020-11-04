import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import pandas as pd
import cv2 as cv
import os
import collections
import scipy.signal as sig


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    if div == 0:
        return [None, None] ## raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class Track:

    def __init__(self):
        self.points = []
        self.last_point = []
        self.last_direc = []
        self.last_size = [1000000, 1000000]
        self.last_imnum = -1


    def add_point(self, point, im_num, roi_size):
        self.points.append(point)
        if len(self.last_point) > 0:
            self.last_direc = np.asarray(self.last_point) - np.asarray(point)
        else:
            self.last_direc = np.asarray([0, 0])
        self.last_point = point
        self.last_imnum = im_num
        self.scale = (roi_size[0]*roi_size[1])/(self.last_size[0]*self.last_size[1])
        self.last_size = roi_size


    def intersection_warning(self, bottom_line):
        sz_thresh = 5
        scale_thresh = 1.00
        ego_direction_ratio = 0.0
        if np.linalg.norm(self.last_direc) < sz_thresh:
            return False, []
        if self.scale < scale_thresh:
            return False, []

        next_point = self.last_point + 100000 * self.last_direc
        inter = line_intersection([self.last_point, next_point], bottom_line)
        left_rafter = inter[1] > bottom_line[1][1] * ego_direction_ratio
        right_rafter = inter[1] < bottom_line[1][1] * (1-ego_direction_ratio)
        if inter[1] != None and left_rafter and right_rafter:
            return True, inter
        else:
            return False, []


class Tracker:

    def __init__(self):
        self.tracks = []
        self.last_image = []
        self.global_disparity = [0, 0]

    def add_point(self, point, im_num, roi_size):
        thresh = 100
        mn_dist = 10000
        for track in self.tracks:
            if track.last_imnum == im_num:
                continue
            dist = np.linalg.norm(np.asarray(track.last_point) - np.asarray(point))
            if dist < mn_dist:
                mn_dist = dist
                mn_dist_track = track
        if mn_dist < thresh:
            mn_dist_track.add_point(point, im_num, roi_size)
        else:
            new_track = Track()
            new_track.add_point(point, im_num, roi_size)
            self.tracks.append(new_track)

    def display(self, im, im_num):
        plt.clf()
        plt.imshow(im[:, :, ::-1])
        plt.title(str(im_num))
        for track in self.tracks:
            if not track.last_imnum == im_num:
                continue
            curr_point = track.last_point
            last_direc = -track.last_direc
            last_point = track.last_direc + curr_point

            # im = cv.arrowedLine(im,curr_point, last_point, [0, 0, 255],3)
            plt.arrow(last_point[0], last_point[1], last_direc[0], last_direc[1], hold=True, color=(0, 1, 0))
        return im

    def intersection_warning(self, bottom_line, im_num, display_inter=False):
        inter = False
        for track in self.tracks:
            if not im_num == track.last_imnum:
                continue
            intersection_warning, intersect = track.intersection_warning(bottom_line)
            if intersection_warning:
                if display_inter:
                    plt.plot([track.last_point[0], intersect[1]], [track.last_point[1], intersect[0]], 'r-')
                    # plt.plot(intersect[1], intersect[0], 'r*')
                inter = True
        return inter

    def track_image(self, im, detections, im_num):
        if len(self.last_image) > 0:
            disparity = find_global_move(im, self.last_image)
            self.global_disparity += disparity
            im = np.roll(im, -self.global_disparity.astype(int))

        self.last_image = im.copy()

        for detection in detections:
            # type = detection[0]
            roi = np.asarray(detection[2])
            cen = [roi[0], roi[1]]
            roi_size = np.abs([roi[2] - roi[0], roi[3] - roi[1]])
            self.add_point(cen, im_num, roi_size)

            roi_tl = [roi[0] - roi[2] / 2, roi[1] - roi[3] / 2, roi[2], roi[3]]
            roi_ = np.asarray(roi_tl).astype(int)

            im = cv.rectangle(img=im, rec=roi_, color=(255, 0, 0), thickness=3)
            return im

def find_global_move(im1, im2):
    im1_gray = np.mean(im1.astype('float'), axis=2)
    im2_gray = np.mean(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
    # calculate the correlation image; note the flipping of onw of the images
    corr_img = sig.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')
    mx = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    shift = np.asarray(im1_gray.shape).astype(float) / 2 - mx
    print(shift)
    return shift


def track_objects(obj , folder):
    tracker = Tracker()
    first_im = True
    fig, ax = plt.subplots(1)
    for im_num in obj.keys():
        if int(im_num) < 10:
            continue
        data = obj[im_num]
        file = data['fileName'][12:]
        file_path = os.path.join(folder, file)
        file_path = file_path[:-4] + ".png"
        if not os.path.exists(file_path):
            continue

        im = cv.imread(file_path)
        bottom_line = [[0., im.shape[0]], [im.shape[1], im.shape[0]]]
        detections = data['detections']
        im = tracker.track_image(im, detections, im_num)

        tracker.display(im[:, :, ::-1], im_num)
        if first_im:
            first_im = False
        else:
            inter = tracker.intersection_warning(bottom_line, im_num, True)
            if inter:
                rect = patches.Rectangle([0, 0], im.shape[1], im.shape[0], 0, linewidth=5,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        # plt.plot(im)
        plt.pause(.0001)
        # out_path = os.path.join(folder, 'warning', file)
        # cv.imwrite(out_path, im)

        a=1

if __name__ == '__main__':
    # folder = r'E:\rafi\got_your_back\data\results_files\res\temp_dir - Copy (9)'
    folder = r'E:\rafi\got_your_back\data\results_files\res\temp_dir - Copy (4)\cut'
    file_path = r"E:\rafi\got_your_back\data\results_files\res\temp_dir - Copy (4)\YoloV3_res\res_pkl.pkl"
    obj = pd.read_pickle(file_path)
    obj = collections.OrderedDict(sorted(obj.items()))
    track_objects(obj, folder)
