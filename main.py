#!/bin/python3

from pathlib import Path

import argparse
import configparser

import signal
import os


from lib.calibration import Calibration
from lib.stereo import StereoCapture

LOAD_DIR = str(Path.home()) + '/.stereo_calibration/'

parser = argparse.ArgumentParser(description='Depth mapping module')
parser.add_argument('-m', '--matcher', default='cuda',
                    help='Matcher to use. Options: stereobm, stereosgbm, cuda')

capture = None

def signal_handler(sig, frame):
    capture.stop()

def main():
    global capture
    signal.signal(signal.SIGINT, signal_handler)

    config = configparser.ConfigParser()
    config.read('settings.conf')

    print('changing permission for /dev/ttyUSB0')
    os.system('sudo chmod a+rw /dev/ttyUSB0 &')
    print('disabling camera auto setting')
    os.system('./disable_auto_camera1.sh &')

    calibrator = Calibration(config, LOAD_DIR)
    if not calibrator.has_calibration():
        print('Calibration required!')
        input('run `python3 calibrate.py`')

        return

    print('Processing stereo images')

    args = parser.parse_args()

    capture = StereoCapture(config, calibrator, args.matcher)
    capture.produce_depth_map()

if __name__ == '__main__':
    main()