# Kalman filters

![title](https://user-images.githubusercontent.com/35414366/132179092-39c96572-5c0e-4b02-85a3-2fe71d394dd8.png)


This repository contains many trials of the author's attempt at coding Kalman filter applications that are not easily attainable elsewhere.

The applications shown in this repository are of reparameterization of initial mean and variance, dual estimation mode, and adaptive estimation of noise covariances, namely the transition noise (Q) and the measurement noise (R).

## Motivation

For a Kalman filter and its relatives, the major issue that arises is effective inference on initial conditions. These initial conditions are 

- Initial state mean and variance

- Q

- R

Thankfully, the formulation of inference for the prototypical Kalman filter is somewhat easier to find (see Pattern Recogntion and Machine Learning by Bishop, Ch13.3.2).

However, they are not as clear-cut for Extended Kalman filter (EKF) or Unscented Kalman filter (UKF) where analytical gradient is harder to find. There also exists the need for adaptive noise estimation whether it be for real-time estimation of variables or for domain-specific purposes (e.g., a biological process which requires a more continous adaptation rather than a likelihood maximizing initial guess).

## Contents

Many scripts pertain to UKF, which aims to solve the issue of computing nonlinear dynamics in systems with latent (hidden) variables.

Example use cases of codes are shown in folder <code>notebooks</code>, one for adaptive UKF and one for dual UKF.

Both <code>/notebooks/UKF_example.ipynb</code> and <code>/notebooks/DUKF_example.ipynb</code> represent a particular use case for the author, and are not meant to be construed as a model solution.

Two of the AUKF scripts, <code>/KF/Schwarz_AUKF.py</code> and <code>/KF/Zhang_AUKF.py</code> are either very unstable or non-functioning. It can reflect the fact that my code is bad or wrong (which isn't unlikely) or that the data isn't one the author envisioned.

Also note that the base <code>UKF.py</code> script was heavily adapted from GitHub profile rlabbe, for whose work can be found at [here](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python "here").


All filters are defined as classes that can be used together utilizing additions as *mixins*. Functionally, it is quite similar to multi-inheritance. It is however different in that mixins are not meant to be initialized independently as a standalone class.

```python
from KF.UKF import UnscentedKalmanFilter # only function that can be used as a standalone filter.
from KF.AUKF import AdaptiveUnscentedKalmanFilter
from KF.DKF import stateKalmanFilter, parameterKalmanFilter
from KF.utils import MixedClassMeta

class sAUKF(
	UnscentedKalmanFilter,
	AdaptiveUnscentedKalmanFilter,
	stateKalmanFilter,
	metaclass=MixedClassMeta
	):
	def __init__(self,*args,**kwargs): pass
class pAUKF(
	UnscentedKalmanFilter,
	AdaptiveUnscentedKalmanFilter,
	parameterKalmanFilter,
	metaclass=MixedClassMeta
	):
	def __init__(self,*args,**kwargs): pass
stateAUKF = sAUKF(*args,**kwargs)
parameterAUKF = pAUKF(*args,**kwargs)
```

Note that the user is free to mix and match, take out the adaptive portion or use another adaptive filter in place. As can be seen in the above code, <code>UnscentedKalmanFilter</code> is the only base class that can be initialized as a standalone filter. The hierarchy of mixins are <code>class(1<2<3<...)</code>. As such, variables initialized in its base are available immediately to the next.


## System Requirements

Tested on MacBook Pro 16inch (2019, Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz). Each runs of filters should take no more than a few dozen seconds at most.

Conda environment .yml is provided directly in the repo. Create an environment in your machine by the following line in the terminal.

<code> conda env create -n environment_name -f environment.yml </code>

## References

[1] E. A. Wan and R. V. D. Merwe, “The unscented Kalman filter for nonlinear estimation,” in Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373), Oct. 2000, pp. 153–158. https://doi.org/10.1109/ASSPCC.2000.882463.

&nbsp;&nbsp;&nbsp; An UKF paper widely cited. The full PhD thesis by R. van der Merwe can be found [here](https://scholararchive.ohsu.edu/downloads/rf55z768s?locale=en "original")

[2] Mehra, R. “Approaches to Adaptive Filtering.” IEEE Transactions on Automatic Control 17, no. 5 (October 1972): 693–98. https://doi.org/10.1109/TAC.1972.1100100.

&nbsp;&nbsp;&nbsp; A genealogical review of adaptive Kalman filters.

[3] Zheng, Binqi, Pengcheng Fu, Baoqing Li, and Xiaobing Yuan. “A Robust Adaptive Unscented Kalman Filter for Nonlinear Estimation with Uncertain Noise Covariance.” Sensors (Basel, Switzerland) 18, no. 3 (March 7, 2018). https://doi.org/10.3390/s18030808.

&nbsp;&nbsp;&nbsp; A fault-detection adaptive UKF aiming to adjust both Q and R. Adjusts both Q and R if criteria is met (over chi-squared threshold). Found in this repo, <code>/KF/Zheng_et_al.py</code>.

[4] Hou, Jing, Yan Yang, He He, and Tian Gao. “Adaptive Dual Extended Kalman Filter Based on Variational Bayesian Approximation for Joint Estimation of Lithium-Ion Battery State of Charge and Model Parameters.” Applied Sciences 9, no. 9 (January 2019): 1726. https://doi.org/10.3390/app9091726.

&nbsp;&nbsp;&nbsp; A dual adaptive UKF (DAUKF) using hyperparameter to adjust R. Similar in approach to [3] but computes R as an inverse-gamma distribution with hyperparameters alpha and beta, estimated through variational bayes.

[5] A. H. Mohamed and K. P. Schwarz, “Adaptive Kalman Filtering for INS/GPS,” Journal of Geodesy, vol. 73, no. 4, pp. 193–203, May 1999, https://doi.org/10.1007/s001900050236.

&nbsp;&nbsp;&nbsp; An adaptive Kalman filter aiming to adjust both Q and R. Uses *correction* sequences to adjust Q and residual sequences to adjust R. Q is not guaranteed to be PD, leading to instability in the filter. Found in this repo, <code>/KF/Schwarz_AUKF.py</code>. Note that the idea was adapted for use in UKF, as the author developed the algorithm for use in canonical Kalman filters. Currently non-functioning.

[6] S. Zhang, “An Adaptive Unscented Kalman Filter for Dead Reckoning Systems,” in 2009 International Conference on Information Engineering and Computer Science, Dec. 2009, pp. 1–4. https://doi.org/10.1109/ICIECS.2009.5365064.

&nbsp;&nbsp;&nbsp; An adaptive UKF aiming to adjust both Q and R. Uses *innnovation* sequences to adjust both Q and R. Both Q and R are not guaranteed to be PD, leading to instability in the filter. Degree to which adaptation occurs is adjusted based on the number of iteration through forgetting scale. Assumes that nonlinearity isn't well captured through EKF or UKF, requiring the filter to adjust more aggressively. The idea is similar to usage of fading memory in typical application of Kalman filters. Found in this repo, <code>/KF/Zhang_AUKF.py</code>.
