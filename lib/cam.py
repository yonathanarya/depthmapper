import cv2

def open_capture(id):
    """
    This method will open camera through cv2.VideoCapture
        args:
            id: camera id from /dev/video*
        return: camera capture
    """
    capture = cv2.VideoCapture(int(id))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    if not capture.isOpened():
        raise Exception("Could not open video device " + str(id))

    return capture