import cv2
from time import sleep

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)
left.set(cv2.CAP_PROP_FRAME_WIDTH,640)
left.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
right.set(cv2.CAP_PROP_FRAME_WIDTH,640)
right.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
sleep(2)

# left.grab()
# right.grab()
# lret, frame0 = left.retrieve()
# rret, frame1 = right.retrieve()

lret, frame0 = left.read()
rret, frame1 = right.read()

sleep(2)

if(lret and rret):
    print("generate left")
    cv2.imwrite('left.png', frame0)
    print("generate right")
    cv2.imwrite('right.png', frame1)