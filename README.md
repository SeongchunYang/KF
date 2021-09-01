# Kalman filters

This repository contains many trials of the author's attempt at coding Kalman filter applications that are not easily attainable elsewhere.

The codes are **NOT** meant to be mathematically bulletproof; in fact, some of the examples such as adaptive filters were chosen specifically for readability and ease of use.

(i.e., post-hoc adaptive measurement noise rather than in-trial adaptation which requires re-compute of the updated mean and variance of variables at each time iteration.)

However, they are not written willy-nilly. Citations are wirtten in the scripts where necessary to justify why it was chosen and where the formulations come from.

Note that the base *UKF.py* script was heavily adapted from GitHub profile rlabbe, for whose work can be found at [here](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python "here").

## System Requirements

Tested on MacBook Pro 16inch (2019), Intel version. Each runs of filters should take no more than a few dozen seconds at most.

Python=3.9

## References

> A UKF paper widely cited (the thesis by R. van der Merwe)

  >> https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

> Conference paper on UKF by R. van der Merwe

  >> Wan, E. A., and R. Van Der Merwe. “The Unscented Kalman Filter for Nonlinear Estimation.” In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373), 153–58, 2000. https://doi.org/10.1109/ASSPCC.2000.882463.

> Genealogical review of adaptive Kalman filters

  >> Mehra, R. “Approaches to Adaptive Filtering.” IEEE Transactions on Automatic Control 17, no. 5 (October 1972): 693–98. https://doi.org/10.1109/TAC.1972.1100100.

> Dual Adaptive Unscented Kalman filter implementation found in this repo.

  >> Zheng, Binqi, Pengcheng Fu, Baoqing Li, and Xiaobing Yuan. “A Robust Adaptive Unscented Kalman Filter for Nonlinear Estimation with Uncertain Noise Covariance.” Sensors (Basel, Switzerland) 18, no. 3 (March 7, 2018). https://doi.org/10.3390/s18030808.

> Closely related Dual Adaptive Unscented Kalman filter using hyperparameter to adjust measurement noise.

  >> Hou, Jing, Yan Yang, He He, and Tian Gao. “Adaptive Dual Extended Kalman Filter Based on Variational Bayesian Approximation for Joint Estimation of Lithium-Ion Battery State of Charge and Model Parameters.” Applied Sciences 9, no. 9 (January 2019): 1726. https://doi.org/10.3390/app9091726.










