import time
import cv2
import os
import numpy as np
import glob

from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

from lib.cam import open_capture

class Calibration:
    """
    This class provide calibration method for stereo image
    """
    def __init__(self, config, capture_directory = "images/"):
        self.width = int(config["general"]["width"])
        self.height = int(config["general"]["height"])

        self.chessboard_rows = int(config["calibration"]["chessboard_rows"])
        self.chessboard_cols = int(config["calibration"]["chessboard_cols"])
        self.chessboard_size = float(config["calibration"]["chessboard_size"])

        self.left_camera_id = config["general"]["left_camera_id"]
        self.right_camera_id = config["general"]["right_camera_id"]

        self.capture_directory = capture_directory

        if os.path.exists("stereoMap.xml") == False:
            self.active_calibration = None
        else:
            print("hehe...")
            self.active_calibration = "Manual"

        if not os.path.exists(self.capture_directory):
            os.mkdir(self.capture_directory)
            os.mkdir(self.capture_directory + "left")
            os.mkdir(self.capture_directory + "right")

    def has_calibration(self):
        """
        This method will check whether the calibration files already exist
        """
        return self.active_calibration != None

    def capture_images(self):
        """
        This method will capture image from camera and save it to image file on /tmp/stereo/
        """
        print("Calibration :: Capturing frames...")

        leftCapture = open_capture(self.left_camera_id)
        rightCapture = open_capture(self.right_camera_id)

        print("Waiting...")
        time.sleep(1)

        for i in range (0, 30, 1):
            print("Hit any key when you have the chessboard prepared")
            print("Waiting 1s...")
            time.sleep(1)
            calibrator = StereoCalibrator(self.chessboard_rows,  self.chessboard_cols,  self.chessboard_size, (self.width, self.height))
            while True:
                _, frame1 = leftCapture.read()
                _, frame2 = rightCapture.read()
                # _, left = leftCapture.read()
                # _, right = rightCapture.read()
                try:
                    calibrator._get_corners(frame1)
                    calibrator._get_corners(frame2)
                    # self.show_corners(frame1, frame2, calibrator._get_corners(frame1), calibrator._get_corners(frame2))
                except ChessboardNotFoundError as error:
                    print(str(i) + ": " + str(error))
                else:
                    cv2.imwrite(self.capture_directory + "left/img" + str(i) + ".png", frame1)
                    cv2.imwrite(self.capture_directory + "right/img" + str(i) + ".png", frame2)
                    calibrator.add_corners((frame1, frame2), show_results=True)
                    print("Frame " + str(i) + " done")
                    break
        print("Calibration :: Captured.")

    def calib_rectify(self):
        chessboardSize = (self.chessboard_cols,self.chessboard_rows)
        frameSize = (self.width,self.height)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = int(self.chessboard_size*10)
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpointsL = [] # 2d points in image plane.
        imgpointsR = [] # 2d points in image plane.


        imagesLeft = sorted(glob.glob('images/left/*.png'))
        imagesRight = sorted(glob.glob('images/right/*.png'))

        for imgLeft, imgRight in zip(imagesLeft, imagesRight):

            imgL = cv2.imread(imgLeft)
            imgR = cv2.imread(imgRight)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
            retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if retL and retR == True:

                objpoints.append(objp)

                cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                imgpointsL.append(cornersL)

                cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                imgpointsR.append(cornersR)

                # Draw and display the corners
                cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
                cv2.imshow('img left', imgL)
                cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
                cv2.imshow('img right', imgR)
                cv2.waitKey(1000)

        cv2.destroyAllWindows()

        # Undistort image

        retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
        heightL, widthL, channelsL = imgL.shape
        newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

        retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
        heightR, widthR, channelsR = imgR.shape
        newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

        # Rectify image

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

        rectifyScale= 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

        stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
        stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

        print("Saving parameters!")
        cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

        cv_file.write('stereoMapL_x',stereoMapL[0])
        cv_file.write('stereoMapL_y',stereoMapL[1])
        cv_file.write('stereoMapR_x',stereoMapR[0])
        cv_file.write('stereoMapR_y',stereoMapR[1])
        cv_file.release()
        print("Calibration saved as stereoMap.xml")