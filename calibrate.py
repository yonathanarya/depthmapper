#!/bin/python3

from lib.calibration import Calibration
import configparser

def main():
    config = configparser.ConfigParser()
    config.read("settings.conf")

    calibrator = Calibration(config)
    print("Please wait for calibration")

    calibrator.calib_rectify()

if __name__ == "__main__":
    main()