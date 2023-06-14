# pattern_recognition_with_IMUs

This github repo contains the implementations to the semester project "Pattern recognition with an IMU" by Ann-Kristin Bergmann. The project was done at the EPFL and the Swiss Data Science Center.

The code provided examines different approaches to reconstruct the trace of an IMU attached to a pen.
A short overview on what to find in which notebook:

- calibration.ipynb: Implementation of [A robust and easy to implement method for IMU calibration without external equipments](https://ieeexplore.ieee.org/document/6907297) method 
- dead-reckoning_baseline.ipynb: Baseline of the dead-reckoning process (implemented by Johan Berdat), exploration of different filters (Butterworth, Chebyshev Typ 1 & 2 and elliptic filtering) and filtering combinations
- dead_reckoning_kalman_madgwick_mahony.ipynb: A comparison of the three different attitude estimators with the implementation given by [ahrs library](https://ahrs.readthedocs.io/en/latest/index.html)
- dr_kmm_sampling_smoothing: Extension of the previous notebook with equal sampling and comparison of smoothing (moving average) and filtering (butter) the raw signals
- classification.ipynb: An implementation of character recognition based on IMU data with keras
- dead-reckoning-gyr.ipynb: An experimental implementation of dead-reckoning with gyroscope data
- preprocessing.py: preprocessing methods for the sentence dataset
