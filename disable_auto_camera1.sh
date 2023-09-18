#!/bin/bash

while :
do
    v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1
    v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto_priority=0
    v4l2-ctl -d /dev/video0 --set-ctrl=brightness=200
    v4l2-ctl -d /dev/video0 --set-ctrl=contrast=50
    v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=0
    v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=300
    v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto=1
    v4l2-ctl -d /dev/video1 --set-ctrl=brightness=200
    v4l2-ctl -d /dev/video1 --set-ctrl=contrast=50
    v4l2-ctl -d /dev/video1 --set-ctrl=backlight_compensation=0
    v4l2-ctl -d /dev/video1 --set-ctrl=exposure_absolute=300
    sleep 1
done