### Kalman filters

This repository contains many trials of the author's attempt at coding Kalman filter applications that are not easily attainable elsewhere.

The codes are **NOT** meant to be mathematically bulletproof; in fact, some of the examples such as adaptive filters were chosen specifically for readability and ease of use.

(i.e., post-hoc adaptive measurement noise rather than in-trial adaptation which requires re-compute of the updated mean and variance of variables at each time iteration.)

However, they are not written willy-nilly. Citations are wirtten in the scripts where necessary to justify why it was chosen and where the formulations come from.

Note that the base *UKF.py* script was heavily adapted from GitHub profile rlabbe, for whose work can be found at https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.

### System Requirements

Tested on MacBook Pro 16inch (2019), Intel version. Each runs of filters should take no more than a few dozen seconds at most.


