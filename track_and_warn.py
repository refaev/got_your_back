import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import pandas as pd
import cv2 as cv
import os
import collections


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
        self.last_imnum = -1


    def add_point(self, point, im_num):
        self.points.append(point)
        if len(self.last_point) > 0:
            self.last_direc = self.last_point - point
        else:
            self.last_direc = np.asarray([0, 0])
        self.last_point = point
        self.last_imnum = im_num

    def intersection_warning(self, bottom_line):
        sz_thresh = 15
        if np.linalg.norm(self.last_direc) < sz_thresh:
            return False, []

        next_point = self.last_point + 100000 * self.last_direc
        inter = line_intersection([self.last_point, next_point], bottom_line)
        if inter[1] != None and inter[1] > bottom_line[1][1] * 0.25 and  inter[1] < bottom_line[1][1] * 0.75:
            return True, inter
        else:
            return False, []


class Tracker:

    def __init__(self):
        self.tracks = []

    def add_point(self, point, im_num):
        thresh = 100
        mn_dist = 10000
        for track in self.tracks:
            if track.last_imnum == im_num:
                continue
            dist = np.linalg.norm(track.last_point - point)
            if dist < mn_dist:
                mn_dist = dist
                mn_dist_track = track
        if mn_dist < thresh:
            mn_dist_track.add_point(point, im_num)
        else:
            new_track = Track()
            new_track.add_point(point, im_num)
            self.tracks.append(new_track)

    def display(self, im, im_num):
        plt.clf()
        plt.imshow(im)
        plt.title(str(im_num))
        for track in self.tracks:
            curr_point = track.last_point
            last_direc = -track.last_direc
            last_point = track.last_direc + curr_point
            # im = cv.arrowedLine(im,curr_point, last_point, [0, 0, 255],3)
            plt.arrow(last_point[0], last_point[1], last_direc[0], last_direc[1], hold=True)
        plt.pause(0.0001)
        return im

    def intersection_warning(self, bottom_line, display_inter=False):
        for track in self.tracks:
            intersection_warning, intersect = track.intersection_warning(bottom_line)
            if intersection_warning:
                plt.plot([track.last_point[0], intersect[1]], [track.last_point[1], intersect[0]], 'r-')
                plt.plot(intersect[1], intersect[0], 'r*')
                plt.plot(intersect[1], intersect[0], 'r*')
                return True
        return False


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
        im = cv.imread(file_path)
        bottom_line = [ [ im.shape[0], 0], [ im.shape[0], im.shape[1]]]
        detections = data['detections']
        for detection in detections:
            # type = detection[0]
            roi = detection[2]
            cen = np.asarray([roi[0], roi[1]])
            tracker.add_point(cen, im_num)
        tracker.display(im[:, :, ::-1], im_num)
        if first_im:
            first_im = False
        else:
            inter = tracker.intersection_warning(bottom_line, True)
            if inter:
                rect = patches.Rectangle([0, 0], im.shape[1], im.shape[0], 0, linewidth=5,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        # plt.plot(im)
        # plt.pause(0.0001)
        # out_path = os.path.join(folder, 'warning', file)
        # cv.imwrite(out_path, im)
    a=1

if __name__ == '__main__':
    folder = r'E:\rafi\got_your_back\data\results_files\res\temp_dir - Copy (9)'
    file_path = os.path.join(folder, r"YoloV3_res\res_pkl.pkl")
    obj = pd.read_pickle(file_path)
    obj = collections.OrderedDict(sorted(obj.items()))
    track_objects(obj, folder)
