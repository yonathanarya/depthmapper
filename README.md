# Jetson Nano CUDA Depthmapper for AGV

This is for my Undergraduate Final Project.
Using 2 Logitech C270 camera as the sensor and Jetson Nano 4GB as the depth processor
The robot controller used in this project is Pixhawk 2.1 Cube.

### Hardware Setup
Laptop: Huawei Matebook X Pro (2018) i7-8550u 16GB RAM

SBC: Nvidia Jetson Nano Development Kit 4GB

Flight (Robot) Controller: Pixhawk 2.1 Cube

GPS: Here3 (u-blox M8P)

Stereo Camera: 2 pcs Logitech C270 HD Webcam

RC Kit: Offbrand offroad 4WD frame kit

Motor: Brushed

ESC: Brushed ESC

Battery: 2S LiPo battery

BEC: UBEC 5V/3A (for powering Jetson)

### Pre-requisites
To use the CUDA version of OpenCV in this project, we have to build our own OpenCV as, by default, the built-in OpenCV shipped with Jetson Nano Jetpack is not having Cuda support.

### Running the program
If running for the first time, prepare the chessboard then run the calibration first:
```
python3 capture_calib.py
```
then
```
python3 calibrate.py
```


How to run this program

Open separate terminal or ssh connection for these:

for broadcasting mavlink connection over UDP to control and monitor pixhawk over wi-fi
```
sudo ./mav.sh
```
to disable camera auto adjust:
```
./disable_auto_camera.sh
```
to run the main program:

```
python3 main.py
```

You can choose which algorithm is used via the `-m` command line switch:

```
python main.py -m <algorithm>

options:

-m stereobm
-m stereosgbm
-m cuda
```

Cuda is the default if this option is ignored.


### Configuration

All configuration will be done via a config file named `settings.conf`. I've made sure to write comments in the default configuration file to give an understanding of what each parameter does.

This file must live next to `main.py`, and will be automatically loaded.


### Output 

Output is the average sum from the disparity map's center area and is sent to Pixhawk.

This is a sample of the result from the disparity map; on the top left corner is FPS measured (when not streamed through SSH, the FPS increases to around 20)
![alt text](https://github.com/yonathanarya/depthmapper/blob/master/sample.png?raw=true)


### Camera delay

Because of the limitation of the USB camera, there are slight delays between the capture of left and right images. This will affect depth mapping, especially when the rover is moving. Here is the delay proof
![alt text](https://github.com/yonathanarya/depthmapper/blob/master/left.png?raw=true)![alt text](https://github.com/yonathanarya/depthmapper/blob/master/right.png?raw=true)
