# depthmapper

This is a (relatively) small utility to be used as a testbed for experimenting with stereo depth mapping in OpenCV.

I built this to find how well depth mapping works on the Nvidia Jetson Nano. Support is present for StereoBM and StereoSGBM, as well as experimental support for AANet+. 

Hardware-wise, this utility has been used with the WaveShare IMX219-83 stereo camera.

Python 3.6 or higher is needed.


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