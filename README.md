# Kalman filters

![title](https://user-images.githubusercontent.com/35414366/132179092-39c96572-5c0e-4b02-85a3-2fe71d394dd8.png)


This repository contains many trials of the author's attempt at coding Kalman filter applications that are not easily attainable elsewhere.

The codes are **NOT** meant to be mathematically bulletproof; in fact, some of the examples such as adaptive filters were chosen specifically for readability and ease of use.

(i.e., post-hoc adaptive measurement noise rather than in-trial adaptation which requires re-compute of the updated mean and variance of variables at each time iteration.)

However, they are not written willy-nilly. Citations are wirtten in the scripts where necessary to justify why it was chosen and where the formulations come from.

Note that the base <code>UKF.py</code> script was heavily adapted from GitHub profile rlabbe, for whose work can be found at [here](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python "here").



## System Requirements

Tested on MacBook Pro 16inch (2019, Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz). Each runs of filters should take no more than a few dozen seconds at most.

Conda environment .yml is provided directly in the repo. Create an environment in your machine by the following line in the terminal.

<code> conda env create -n environment_name -f environment.yml </code>

## References

[1] E. A. Wan and R. V. D. Merwe, “The unscented Kalman filter for nonlinear estimation,” in Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373), Oct. 2000, pp. 153–158. https://doi.org/10.1109/ASSPCC.2000.882463.

> An UKF paper widely cited. The full PhD thesis by R. van der Merwe can be found [here](https://scholararchive.ohsu.edu/downloads/rf55z768s?locale=en "original")

[2] Mehra, R. “Approaches to Adaptive Filtering.” IEEE Transactions on Automatic Control 17, no. 5 (October 1972): 693–98. https://doi.org/10.1109/TAC.1972.1100100.

> A genealogical review of adaptive Kalman filters.

[3] Zheng, Binqi, Pengcheng Fu, Baoqing Li, and Xiaobing Yuan. “A Robust Adaptive Unscented Kalman Filter for Nonlinear Estimation with Uncertain Noise Covariance.” Sensors (Basel, Switzerland) 18, no. 3 (March 7, 2018). https://doi.org/10.3390/s18030808.

> Dual Adaptive Unscented Kalman filter implementation found in this repo.

[4] Hou, Jing, Yan Yang, He He, and Tian Gao. “Adaptive Dual Extended Kalman Filter Based on Variational Bayesian Approximation for Joint Estimation of Lithium-Ion Battery State of Charge and Model Parameters.” Applied Sciences 9, no. 9 (January 2019): 1726. https://doi.org/10.3390/app9091726.

> Closely related Dual Adaptive Unscented Kalman filter using hyperparameter to adjust measurement noise.










