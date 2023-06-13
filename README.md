# pattern_recognition_with_IMUs

This github repo contains the implementations to the semester project "Pattern recognition with an IMU" by Ann-Kristin Bergmann. The project was done at the EPFL and the Swiss Data Science Center.

The code provided examines different approaches to reconstruct the trace of an IMU attached to a pen.
A short overview on what to find in which notebook:

- calibration.ipynb: Implementation of [A robust and easy to implement method for IMU calibration without external equipments](https://ieeexplore.ieee.org/document/6907297) method 
- dead_reckoning_kalman_madgwick_mahony.ipynb: A comparison of the three different attitude estimators with the implementation given by [ahrs library](https://ahrs.readthedocs.io/en/latest/index.html)
