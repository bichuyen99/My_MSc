# Overview

Goal: find locations of a given item on a store shelf. \
The output should be an array of locations [(x_min, y_min, width, height), ...]. \
Values are in the scale 0-1 (yolo format). (x_min, y_min) is left-upper corner of a bounding box, x_min = x_min *in pixels* / image_width.
