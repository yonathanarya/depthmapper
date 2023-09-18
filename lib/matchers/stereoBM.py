import cv2

class StereoBMMatcher:
    def __init__(self, config):
        self.width = int(config['general']['width'])
        self.height = int(config['general']['height'])

        downsample = int(config['general']['downsample_factor'])

        self.resize = (int(self.width / downsample), int(self.height / downsample))

        # Apply configuration settings
        self.sbm = cv2.StereoBM_create(
            numDisparities=int(config['stereobm']['num_disparities']),
            blockSize=int(config['stereobm']['block_size'])
        )
        self.sbm.setPreFilterType(1)
        self.sbm.setMinDisparity(int(config['stereobm']['min_disparity']))
        self.sbm.setNumDisparities(int(config['stereobm']['num_disparities']))
        self.sbm.setTextureThreshold(int(config['stereobm']['texture_threshold']))
        self.sbm.setUniquenessRatio(int(config['stereobm']['uniqueness_ratio']))
        self.sbm.setSpeckleRange(int(config['stereobm']['speckle_range']))
        self.sbm.setSpeckleWindowSize(int(config['stereobm']['speckle_window']))

    def process_pair(self, rectified_pair):
        left = cv2.resize(rectified_pair[0], self.resize)
        right = cv2.resize(rectified_pair[1], self.resize)

        disparity = self.sbm.compute(left, right)

        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))
        print("using BM")

        return disparity