#!/bin/python3

from pathlib import Path

from lib.calibration import Calibration
import configparser

LOAD_DIR = str(Path.home()) + "/.stereo_calibration/"

def main():
    config = configparser.ConfigParser()
    config.read("settings.conf")

    calibrator = Calibration(config, LOAD_DIR)
    input("Press any key if calibration image is prepared")

    calibrator.capture_images()

if __name__ == "__main__":
    main()