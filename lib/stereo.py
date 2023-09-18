import cv2
import sys
import numpy as np
from lib.helpers import open_capture
import time

from lib.matchers.stereoBM import StereoBMMatcher
from lib.matchers.cuda import cudaMatcher
from lib.matchers.stereoSGBM import StereoSGBMMatcher

class StereoCapture:
    def __init__(self, config, calibrator, matcher = 'stereobm'):
        self.config = config
        self.calibrator = calibrator
        self.stopped = False

        self.width = int(config['general']['width'])
        self.height = int(config['general']['height'])

        self.left_camera_id = int(config['general']['left_camera_id'])
        self.right_camera_id = int(config['general']['right_camera_id'])

        self.show_rgb = int(config['general']['show_rgb_frame'])

        print(matcher)

        if matcher == 'stereobm':
            self.matcher = StereoBMMatcher(config)
        elif matcher == 'cuda':
            self.matcher = cudaMatcher(config)
        elif matcher == 'stereosgbm':
            self.matcher = StereoSGBMMatcher(config)
        else:
            print("unknown matcher specified, stopping")
            sys.exit()

    def produce_depth_map(self):
        self.leftCapture = open_capture(self.left_camera_id)
        self.rightCapture = open_capture(self.right_camera_id)

        cv2.namedWindow("Depth map")

        ticks = []
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0

        while self.stopped == False:
            start = int(time.time() * 1000.0)
            end = 0

            left_grabbed, left_frame = self.leftCapture.read()
            right_grabbed, right_frame = self.rightCapture.read()
            
            if left_grabbed and right_grabbed:
                rectified_pair = self.rectify(left_frame, right_frame)
                # rectified_pair = [cv2.cvtColor(cv2.imread("left.jpg"), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread("right.jpg"), cv2.COLOR_BGR2GRAY)]
                disparity = self.matcher.process_pair(rectified_pair)

                end = int(time.time() * 1000.0)

                disparity_normal = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                image = np.array(disparity_normal, dtype = np.uint8)
                disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
                # font which we will be using to display FPS
                font = cv2.FONT_HERSHEY_SIMPLEX
                # time when we finish processing for this frame
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
            
                fps = float(fps)
                fpsscreen = str(round(fps,3))
            
                # converting the fps to string so that we can display it on frame
                # by using putText function
                fps = str(fps)
                print("FPS = " , fps)
                # putting the FPS count on the frame
                cv2.putText(disparity_color, fpsscreen, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                # Show depth map
                if self.show_rgb:
                    cv2.imshow("Depth map", np.hstack((self.rectify(left_frame, right_frame))))
                else:
                    cv2.imshow("Depth map", disparity_color)
                    # continue

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
            else:
                end = int(time.time() * 1000.0)

            ticks.append(end - start)

        # Log out timings
        minval = min(ticks)
        maxval = max(ticks)
        avgval = np.mean(ticks)

        print('Timings -- min: ' + str(minval) + 'ms, max: ' + str(maxval) + 'ms, mean: ' + str(avgval) + 'ms')

    def rectify(self, left_frame, right_frame):
        # Convert to greyscale
        left_grey = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Apply rectification
        return self.calibrator.rectify(left_grey, right_grey)

    def stop(self):
        self.leftCapture.stop()
        self.rightCapture.stop()

        self.stopped = True

