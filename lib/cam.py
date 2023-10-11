import cv2
import configparser

def open_capture(id):
    """
    This method will open camera through cv2.VideoCapture
        args:
            id: camera id from /dev/video*
        return: camera capture
    """
    config = configparser.ConfigParser()
    config.read("settings.conf")
    width = int(config["general"]["width"])
    height = int(config["general"]["height"])

    capture = cv2.VideoCapture(int(id))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    if not capture.isOpened():
        raise Exception("Could not open video device " + str(id))

    return capture