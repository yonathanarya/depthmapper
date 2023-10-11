import cv2

class StereoSGBMMatcher:
    """
    This class will process the image input into disparity map using Stereo Semi Global Block Matching method
    """
    def __init__(self, config):
        self.width = int(config["general"]["width"])
        self.height = int(config["general"]["height"])
        blockSize = int(config["stereosgbm"]["block_size"])

        # Apply configuration settings
        self.sbm = cv2.StereoSGBM_create(
            int(config["stereosgbm"]["min_disparity"]),
            int(config["stereosgbm"]["num_disparities"]),
            blockSize,
            int(config["stereosgbm"]["p1_factor"]) * blockSize * blockSize,
            int(config["stereosgbm"]["p2_factor"]) * blockSize * blockSize,
            int(config["stereosgbm"]["disp_12_max_diff"]),
            int(config["stereosgbm"]["prefilter_cap"]),
            int(config["stereosgbm"]["uniqueness_ratio"]),
            int(config["stereosgbm"]["speckle_window"]),
            int(config["stereosgbm"]["speckle_range"])
            )

    def process_pair(self, rectified_pair):
        """
        Computes the disparity map using the named algorithm.
        Args:
            rectified_pair: array of left and right image that already rectified/calibrated
        Returns:
            The disparity map
        """
        left = rectified_pair[0]
        right = rectified_pair[1]

        disparity = self.sbm.compute(left, right)

        disparity = cv2.erode(disparity, None, iterations=1)
        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))
        print("using SGBM")

        return disparity