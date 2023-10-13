#!/bin/python3

from pathlib import Path

from lib.calibration import Calibration
import configparser

LOAD_DIR = str(Path.home()) + "/.stereo_calibration/"

def main():
    config = configparser.ConfigParser()
    config.read("settings.conf")

    calibrator = Calibration(config, LOAD_DIR)
    print("Please wait for calibration")

    calibrator.calib_rectify()

if __name__ == "__main__":
    main()