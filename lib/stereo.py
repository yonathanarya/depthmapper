import cv2
import sys
import numpy as np
from lib.cam import open_capture
import time

from lib.mavlink import sendDepth
from lib.algo.bm import StereoBMMatcher
from lib.algo.cuda import cudaMatcher
from lib.algo.sgbm import StereoSGBMMatcher

class StereoCapture:
    def __init__(self, config, calibrator, matcher = "stereobm"):
        self.config = config
        self.calibrator = calibrator
        self.stopped = False
        self.mav = sendDepth(config)

        self.width = int(config["general"]["width"])
        self.height = int(config["general"]["height"])

        self.left = int(config["estimated_depth"]["left_pixel"])
        self.right = int(config["estimated_depth"]["right_pixel"])
        self.up = int(config["estimated_depth"]["up_pixel"])
        self.down = int(config["estimated_depth"]["down_pixel"])

        self.left_camera_id = int(config["general"]["left_camera_id"])
        self.right_camera_id = int(config["general"]["right_camera_id"])

        self.show_gray = int(config["general"]["show_gray_frame"])
        self.disable_stream = int(config["general"]["disable_stream"])
        self.enable_record = int(config["general"]["enable_record"])
        self.result = cv2.VideoWriter("result.avi", 
                                      cv2.VideoWriter_fourcc(*"MJPG"),
                                      10, (self.width,self.height))

        print(matcher)

        if matcher == "bm":
            self.matcher = StereoBMMatcher(config)
        elif matcher == "cuda":
            self.matcher = cudaMatcher(config)
        elif matcher == "sgbm":
            self.matcher = StereoSGBMMatcher(config)
        else:
            print("unknown matcher specified, stopping")
            sys.exit()

    def produce_depth_map(self):
        """
        This method priduce the depth map in loop and show, record, or save the frame as picture depends on config on settings.conf
        """
        self.leftCapture = open_capture(self.left_camera_id)
        self.rightCapture = open_capture(self.right_camera_id)

        ticks = []
        prev_frame_time = 0
        new_frame_time = 0
        
        while self.stopped == False:
            start = int(time.time() * 1000.0)
            end = 0
            
            self.leftCapture.grab()
            self.rightCapture.grab()
            
            left_grabbed, left_frame = self.leftCapture.retrieve()
            right_grabbed, right_frame = self.rightCapture.retrieve()
            
            try:
                if left_grabbed and right_grabbed:
                    rectified_pair = self.rectify(left_frame, right_frame)
                    # rectified_pair = [cv2.cvtColor(cv2.imread("left.png"), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread("right.png"), cv2.COLOR_BGR2GRAY)]
                    
                    disparity = self.matcher.process_pair(rectified_pair)

                    end = int(time.time() * 1000.0)

                    disparity_normal = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                    image = np.array(disparity_normal, dtype = np.uint8)
                    disparity_color = image

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time
                    fps = float(fps)
                    fpsscreen = str(round(fps,3))
                    fps = str(fps)
                    print("FPS = " , fps)

                    cv2.putText(disparity_color, "FPS:"+fpsscreen, (10, 30), font, 1, (0, 0, 0), 15, cv2.LINE_AA)
                    cv2.putText(disparity_color, "FPS:"+fpsscreen, (10, 30), font, 1, (255, 255, 255), 3, cv2.LINE_AA)

                    if self.show_gray:
                        print("Show grayscale from left and right frame")
                        cv2.imshow("GRAYSCALE", np.hstack((self.rectify(left_frame, right_frame))))
                    elif self.disable_stream:
                        if self.enable_record:
                            self.result.write(cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                            estimated = self.estimate(disparity_color)
                            self.mav.depth(estimated)
                        else:
                            # cv2.imwrite("sample.png", cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                            # cv2.imwrite("orig.png", np.hstack((self.rectify(left_frame, right_frame))))
                            estimated = self.estimate(disparity_color)
                            self.mav.depth(estimated)
                    else:
                        print("DEPTH MAP")
                        cv2.imwrite("sample.png", cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                        cv2.imshow("Depth map", disparity_color)
                        estimated = self.estimate(disparity_color)
                        self.mav.depth(estimated)
                        # continue

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                else:
                    end = int(time.time() * 1000.0)
            except KeyboardInterrupt:
                end = int(time.time() * 1000.0)

            ticks.append(end - start)

        # Log out timings
        minval = min(ticks)
        maxval = max(ticks)
        avgval = np.mean(ticks)

        print("Timings -- min: " + str(minval) + "ms, max: " + str(maxval) + "ms, mean: " + str(avgval) + "ms")

    def rectify(self, left_frame, right_frame):
        """
        This method rectify the GRAY frame of both camera
        args:
            left_frame: left image captured
            right_frame: right image captured
        return: rectified GRAY image
        """
        left_grey = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        return self.calibrator.rectify(left_grey, right_grey)
    
    def estimate(self, disparity):
        """
        This method will compute distance from disparity map that will be sent over mavlink
        args:
            disparity: disparity map of in GRAYSCALE/8 bit color
        return: computed distance
        """
        config = self.config
        factor = factor = float(config["estimated_depth"]["depth_factor"])
        arr = np.array(disparity)
        # depth = float(depth)
        depth = np.sum(arr[self.up:self.down, self.left:self.right])
        count = np.count_nonzero(arr[self.up:self.down, self.left:self.right])
        estimated = 255-(factor*depth/count)
        black = ((self.down-self.up)*(self.right-self.left))-count
        print("estimated depth: " + str(round(estimated,1)))
        print("black pixel: "+str(black))
        if black > 10000:
            output = 100
        else:
            if estimated >= 170 and estimated < 255:
                output = 200
            elif estimated >= 120 and estimated < 170:
                output = 150
            elif estimated <120:
                output = 100
        return int(output)


    def stop(self):
        """
        This method stop the capture of left and right camera
        """
        self.leftCapture.stop()
        self.rightCapture.stop()

        self.stopped = True

