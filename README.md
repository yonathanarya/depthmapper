# depthmapper

This is for my Undergraduate Final Project.
Using 2 Logitech C270 camera as the sensor and Jetson Nano 4GB as the depth processor
The robot controller used in this project is Pixhawk 2.1 Cube

### Pre-requisities
To use the cuda version of OpenCV in this project, we have to build our own OpenCV as by default, the built-in OpenCV shipped with Jetson Nano Jetpack is not having Cuda support.

### Running the utility

Assuming you have a stereo camera board attached and available on `/dev/video0` and `/dev/video1`, you should be able to just run:

```bash
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

StereoBM is the default if this option is omitted.

### Output 

For now, output is the generated disparity map and is shown in a new window. By default, the left hand frame the map corresponds to is shown too.

### Configuration

All configuration is to be done via a config file, named `settings.conf`. I've made sure to write comments in the default configuration file to give an understanding of what each parameter does.

This file needs to live next to `main.py`, and will be automatically loaded.