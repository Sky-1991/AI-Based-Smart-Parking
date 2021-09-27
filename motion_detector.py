import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE
from collections import Counter
import time
import datetime


class MotionDetector:
    LAPLACIAN = 2.5
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.time_log = [self.exit_time() for i in range(len(coordinates))]
        self.centers = [0 for i in range(len(coordinates))]


    
    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]
        mean = np.mean(np.abs(laplacian * self.mask[index]))
        print(mean, MotionDetector.LAPLACIAN)
        status = mean < MotionDetector.LAPLACIAN

        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        # gets called when something gets changed
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]

    @staticmethod
    def exit_time(hr=0, mins=0.5):
        ts = time.time()
        wts = ts + hr * 60 * 60 + mins * 60
        return wts
        # while time.time() < wts:
        #     # dt_object = datetime.fromtimestamp(wts - time.time())
        #     # print(dt_object.time())
        #     print(str(datetime.timedelta(seconds=int(wts - time.time()))))

    @staticmethod
    def waiting_time(t):
        return str(datetime.timedelta(seconds=int(t - time.time())))




class CaptureReadError(Exception):
    pass
