import numpy as np
import cv2
from cv2 import cuda
import configparser

class cudaMatcher:
    """
    This class takes care of the CUDA input such that such that images
    can be provided as numpy array
    """
    def __init__(self, config) -> None:        
        self.stereo_bm_cuda = cuda.createStereoBM(numDisparities=int(config["cuda_bm"]["min_disparity"]),
                                                  blockSize=int(config["cuda_bm"]["block_size"]))
        self.stereo_bp_cuda = cuda.createStereoBeliefPropagation(ndisp=int(config["cuda_bm"]["bp_ndisp"]))
        self.stereo_bcp_cuda = cuda.createStereoConstantSpaceBP(int(config["cuda_bm"]["min_disparity"]))
        self.stereo_sgm_cuda = cuda.createStereoSGM(minDisparity=int(config["cuda_sgm"]["min_disparity"]),
                                                     numDisparities=int(config["cuda_sgm"]["num_disparities"]),
                                                     uniquenessRatio=int(config["cuda_sgm"]["uniqueness_ratio"])
                                                     )
    @staticmethod
    def __numpy_to_gpumat(np_image: np.ndarray) -> cv2.cuda_GpuMat:
        """
        This method converts the numpy image matrix to a matrix that
        can be used by opencv cuda.
        Args:
            np_image: the numpy image matrix
        Returns:
            The image as a cuda matrix
        """
        image_cuda = cv2.cuda_GpuMat()
        image_cuda.upload(np_image)
        return image_cuda
    # def process_pair(self, left_img: np.ndarray,
    #                       right_img: np.ndarray,
    #                       algorithm_name: str = "stereo_sgm_cuda"
    #                       ) -> np.ndarray:
    def process_pair(self, rectified_pair,
                          algorithm_name: str = "stereo_bm_cuda"
                          ) -> np.ndarray:
        """
        Computes the disparity map using the named algorithm.
        Args:
            rectified_pair: array of left and right image that already rectified/calibrated
            algorithm_name: the algorithm to use for calculating the disparity map
        Returns:
            The disparity map
        """
        algorithm = getattr(self, algorithm_name)
        left_cuda = self.__numpy_to_gpumat(rectified_pair[0])
        right_cuda = self.__numpy_to_gpumat(rectified_pair[1])
        # if algorithm_name == "stereo_bm_cuda":
        #     disparity_cuda_2 = cv2.cuda_GpuMat()
        #     disparity_cuda_1 = algorithm.compute(left_cuda, right_cuda, disparity_cuda_2)
        #     print("using cuda...")
        #     return disparity_cuda_1.download()
        # else:
        config = configparser.ConfigParser()
        config.read("settings.conf")
        if(algorithm_name == "stereo_bm_cuda"):
            self.stereo_bm_cuda.setPreFilterType(int(config["cuda_bm"]["prefilter_type"]))
            self.stereo_bm_cuda.setPreFilterType(int(config["cuda_bm"]["prefilter_size"]))
            self.stereo_bm_cuda.setPreFilterType(int(config["cuda_bm"]["prefilter_cap"]))
            self.stereo_bm_cuda.setMinDisparity(int(config["cuda_bm"]["min_disparity"]))
            self.stereo_bm_cuda.setBlockSize(int(config["cuda_bm"]["block_size"]))
            self.stereo_bm_cuda.setSmallerBlockSize(int(config["cuda_bm"]["smaller_block_size"]))
            self.stereo_bm_cuda.setNumDisparities(int(config["cuda_bm"]["num_disparities"]))
            self.stereo_bm_cuda.setTextureThreshold(int(config["cuda_bm"]["texture_threshold"]))
            self.stereo_bm_cuda.setUniquenessRatio(int(config["cuda_bm"]["uniqueness_ratio"]))
            self.stereo_bm_cuda.setSpeckleRange(int(config["cuda_bm"]["speckle_range"]))
            self.stereo_bm_cuda.setSpeckleWindowSize(int(config["cuda_bm"]["speckle_window"]))
            self.stereo_bm_cuda.setDisp12MaxDiff(int(config["cuda_bm"]["disp_diff"]))
            # right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_bm_cuda);
            disparity_cuda_l = algorithm.compute(left_cuda, right_cuda, cv2.cuda_Stream.Null())
            # disparity_cuda_r = algorithm.compute(right_cuda, left_cuda, cv2.cuda_Stream.Null())

            # wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_bm_cuda);
            # wls_filter.setLambda(float(config["filter"]["lmbda"]));
            # wls_filter.setSigmaColor(float(config["filter"]["sigma"]));
            # filtered_disp = wls_filter.filter(disparity_cuda_l.download(), rectified_pair[0], disparity_map_right=disparity_cuda_r.download());
            return disparity_cuda_l.download()
        elif(algorithm_name == "stereo_sgm_cuda"):
            self.stereo_sgm_cuda.setBlockSize(int(config["cuda_sgm"]["block_size"]))
            self.stereo_sgm_cuda.setDisp12MaxDiff(int(config["cuda_sgm"]["disp_12_max_diff"]))
            self.stereo_sgm_cuda.setMinDisparity(int(config["cuda_sgm"]["min_disparity"]))
            self.stereo_sgm_cuda.setNumDisparities(int(config["cuda_sgm"]["num_disparities"]))
            self.stereo_sgm_cuda.setP1(int(config["cuda_sgm"]["p1_factor"]))
            self.stereo_sgm_cuda.setP2(int(config["cuda_sgm"]["p2_factor"]))
            self.stereo_sgm_cuda.setPreFilterCap(int(config["cuda_sgm"]["prefilter_cap"]))
            self.stereo_sgm_cuda.setUniquenessRatio(int(config["cuda_sgm"]["uniqueness_ratio"]))
            self.stereo_sgm_cuda.setSpeckleRange(int(config["cuda_sgm"]["speckle_range"]))
            self.stereo_sgm_cuda.setSpeckleWindowSize(int(config["cuda_sgm"]["speckle_window"]))
            disparity_sgm_cuda_2 = cv2.cuda_GpuMat()
            disparity_sgm_cuda_1 = algorithm.compute(left_cuda,
                                                     right_cuda,
                                                     disparity_sgm_cuda_2)
            return disparity_sgm_cuda_1.download()
        else:
            disparity_cuda = algorithm.compute(left_cuda, right_cuda, cv2.cuda_Stream.Null())
            return disparity_cuda.download()