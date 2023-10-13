import cv2
import sys
import numpy as np
from lib.cam import open_capture
import time
import configparser

from lib.mavlink import sendDepth
from lib.algo.bm import StereoBMMatcher
from lib.algo.cuda import cudaMatcher
from lib.algo.sgbm import StereoSGBMMatcher

class StereoCapture:
    def __init__(self, config, calibrator, matcher = "cudasgm"):
        self.config = config
        self.calibrator = calibrator
        self.stopped = False
        self.mav = sendDepth(config)

        self.width = int(config["general"]["width"])
        self.height = int(config["general"]["height"])

        self.left_camera_id = int(config["general"]["left_camera_id"])
        self.right_camera_id = int(config["general"]["right_camera_id"])

        self.show_gray = int(config["general"]["show_gray_frame"])
        self.disable_stream = int(config["general"]["disable_stream"])
        self.enable_record = int(config["general"]["enable_record"])
        self.result = cv2.VideoWriter("result.avi", 
                                      cv2.VideoWriter_fourcc(*"MJPG"),
                                      10, (self.width,self.height))

        cv_file = cv2.FileStorage()
        cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
        self.stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        self.stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        self.stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        self.stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()    

        print(matcher)

        if matcher == "bm":
            self.matcher = StereoBMMatcher(config)
        elif matcher == "cudabm":
            self.matcher = cudaMatcher(config, "stereo_bm_cuda")
        elif matcher == "cudasgm":
            self.matcher = cudaMatcher(config, "stereo_sgm_cuda")
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

        self.ticks = []
        prev_frame_time = 0
        new_frame_time = 0
        
        while self.stopped == False:
            self.start = int(time.time() * 1000.0)
            self.end = 0
            
            self.leftCapture.grab()
            self.rightCapture.grab()

            config = configparser.ConfigParser()
            config.read("settings.conf")
            
            if int(config['general']['flip']) == 0:
                left_grabbed, left_frame = self.leftCapture.retrieve()
                right_grabbed, right_frame = self.rightCapture.retrieve()
            elif int(config['general']['flip']) == 1:
                left_grabbed, right_frame = self.leftCapture.retrieve()
                right_grabbed, left_frame = self.rightCapture.retrieve()
            elif int(config['general']['flip']) == 3:
                left_grabbed = True
                right_grabbed = True
                left_frame = cv2.imread("2.png")
                right_frame = cv2.imread("1.png")
                rectified_pair = [cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)]
                        
            try:
                if left_grabbed and right_grabbed:                                        
                    if int(config['general']['flip']) == 0:
                        frame_left = cv2.remap(cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY), self.stereoMapL_x, self.stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        frame_right = cv2.remap(cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY), self.stereoMapR_x, self.stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        rectified_pair = [frame_left, frame_right]
                    elif int(config['general']['flip']) == 1:
                        frame_left = cv2.remap(cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY), self.stereoMapR_x, self.stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        frame_right = cv2.remap(cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY), self.stereoMapL_x, self.stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        rectified_pair = [frame_left, frame_right]
                    
                    disparity = self.matcher.process_pair(rectified_pair)

                    self.end = int(time.time() * 1000.0)

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
                        cv2.imshow("GRAYSCALE", np.hstack((right_frame, left_frame)))
                    elif self.disable_stream:
                        if self.enable_record:
                            # cv2.imwrite("sample.png", cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                            # cv2.imwrite("orig.png", np.hstack((self.rectify(left_frame, right_frame))))
                            self.result.write(cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                            estimated = self.estimate(disparity_color)
                            self.mav.depth(estimated)
                        else:
                            cv2.imwrite("sample.png", cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                            cv2.imwrite("orig.png", np.hstack((rectified_pair[1], rectified_pair[0])))
                            cv2.imwrite("left_rectified.png", rectified_pair[0])
                            cv2.imwrite("right_rectified.png", rectified_pair[1])
                            estimated = self.estimate(disparity_color)
                            self.mav.depth(estimated)
                    else:
                        print("DEPTH MAP")
                        # cv2.imwrite("sample.png", cv2.applyColorMap(image, cv2.COLORMAP_BONE))
                        cv2.imshow("Depth map", disparity_color)
                        estimated = self.estimate(disparity_color)
                        self.mav.depth(estimated)
                        # continue

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                else:
                    self.end = int(time.time() * 1000.0)
            except KeyboardInterrupt:
                self.end = int(time.time() * 1000.0)

            self.ticks.append(self.end - self.start)

        # Log out timings
        minval = min(self.ticks)
        maxval = max(self.ticks)
        avgval = np.mean(self.ticks)

        print("Timings -- min: " + str(minval) + "ms, max: " + str(maxval) + "ms, mean: " + str(avgval) + "ms")

    def rectify(self, left_frame, right_frame):
        """
        This method rectify the GRAY frame of both camera
        args:
            left_frame: left image captured
            right_frame: right image captured
        return: rectified GRAY image
        """
        # left_frame = cv2.resize(left_frame,(320,180))
        # right_frame = cv2.resize(right_frame,(320,180))
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

        config = configparser.ConfigParser()
        config.read("settings.conf")
        self.left = int(config["estimated_depth"]["left_pixel"])
        self.right = int(config["estimated_depth"]["right_pixel"])
        self.up = int(config["estimated_depth"]["up_pixel"])
        self.down = int(config["estimated_depth"]["down_pixel"])

        factor = float(config["estimated_depth"]["depth_factor"])
        arr = np.array(disparity)
        # depth = float(depth)
        depth = np.sum(arr[self.up:self.down, self.left:self.right])
        ## Zero value is ignored because it usually the shadow result
        count = np.count_nonzero(arr[self.up:self.down, self.left:self.right])
        # estimated = 255-(factor*depth/count)
        estimated = (factor*depth/count)
        black = ((self.down-self.up)*(self.right-self.left))-count
        print("estimated depth: " + str(round(estimated,1)))
        print("black pixel: "+str(black))
        if estimated >= float(config["estimated_depth"]["max_depth"]) and estimated <= 255:
            if black > float(config["estimated_depth"]["black_pixel"]):
                proximity = 80
            else:
                proximity = 200
        elif estimated >= float(config["estimated_depth"]["min_depth"]) and estimated < float(config["estimated_depth"]["max_depth"]):
            if black > float(config["estimated_depth"]["black_pixel"]):
                proximity = 80
            else:
                proximity = 150
        elif estimated < float(config["estimated_depth"]["min_depth"]):
            if black > float(config["estimated_depth"]["black_pixel"]):
                proximity = 80
            else:
                proximity = 50
        else:
            proximity = 0
        return int(proximity)


    def stop(self):
        """
        This method stop the capture of left and right camera and show timing log
        """

        # Log out timings
        minval = min(self.ticks)
        maxval = max(self.ticks)
        avgval = np.mean(self.ticks)

        print("Timings -- min: " + str(minval) + "ms, max: " + str(maxval) + "ms, mean: " + str(avgval) + "ms")

        self.leftCapture.stop()
        self.rightCapture.stop()

        self.stopped = True

