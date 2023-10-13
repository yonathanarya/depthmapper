#!/bin/python3

from lib.calibration import Calibration
import configparser

def main():
    config = configparser.ConfigParser()
    config.read("settings.conf")

    calibrator = Calibration(config)
    input("Press any key if calibration image is prepared")

    calibrator.capture_images()

if __name__ == "__main__":
    main()