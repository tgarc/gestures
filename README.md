# gesture_recognition
A homebrew hand gesture recognition suite.

I started this project so I could eventually implement it on a beaglebone to
control a [pandora](http://www.pandora.com/) client through
[pianobar](http://6xq.net/projects/pianobar/) using swiping gestures.


## Overview
I've split the project up as follows

* Motion detection/segmentation
  * Detects motion to help with hand detection and tracking
* Skin segmentation
  * Segments skin color pixels for hand detection and segmentation
* Gesture classifier
  * Matches recorded hand gestures with best match from a stored database of
    gestures


## Motion detection
Uses ideas from [A System for Video Surveillance and
Monitoring](https://www.ri.cmu.edu/pub_files/pub2/collins_robert_2000_1/collins_robert_2000_1.pdf)
to detect and segment transient moving objects in scene.


## Skin segmentation
Uses CrCb thresholding based on statistics collected from datasets of skin
pixels.


## Gesture classifier
A slight modification of the [$1 classifier](http://depts.washington.edu/aimgroup/proj/dollar/) by Wobbrock et al. - a geometric template matcher.


## Dependencies
* python 2.7
* matplotlib
* opencv
* h5py (for data analysis)
