# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# As close of an exact implementation of doi: 10.1109/ICIECS.2009.5365064 (Zhang et al., IEEE, 2009).
# 2.
# Reason behind the name 'forgetting scale' parameter is the author's motivation behind this implemnetation,
# which is that nonlinearity cause errors (untracked values) that accumulate over time in the filter [1].
# This is said to be compensated by the every increasing 'd' parameter calculated at every iteration
# which makes the current estimate of error dominate the overall evaluation of Q and R as time passes.

# [1]
# Technically, this isn't a unique problem to just nonlinear filters. Even canonical KFs may suffer the
# same fate if the dynamics aren't well captured through the transition function (fx) and the observation
# function (hx). As such, a fading memory filter exists which increases the predictive covariance slightly
# to mitigate the effect of the past in the filter at every iteration.

import numpy as np
from numpy import dot
from copy import copy, deepcopy
from KF.UKF import UnscentedKalmanFilter

class adaptiveUKF(UnscentedKalmanFilter):
    def __init__(
        self, 
        dim_z, 
        dim_x, 
        z0,
        P0,
        fx, 
        hx, 
        points_fn, 
        Q, 
        R, 
        b
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.b = b # forgetting scale (0<b<1)

    def adapt_QR(self, i, x, **kwargs):
        '''
        As is evident, both Q and R adjustments are made using innovation.
        The formulation below makes it so that we can't sufficiently guarantee that this will be PD.
        Further adaption in future for this is required for stability.
        '''
        self.d = (1-self.b)/(1-self.b**(i))
        self.R = self.R - self.d * (np.outer(self.innovation, self.innovation) - self.Pxx_c_p)
        self.Q = self.Q + self.d * (
            np.outer(
                dot(self.K, self.innovation),
                dot(self.K, self.innovation)
            ) + self.P - self.P_c_p
        )
    
    def correct_update(self, x, **kwargs):
        # recompute mean and variance per updated Q
        self.z_c_c, self.Pzz_c_c = self.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cv = self.Q
        )
        # recompute mean and variance per updated R
        self.x_c_c, self.Pxx_c_c = self.UT(
            sigmas = self.sigmas_h_c_p,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        hphi_c_c = []
        for s in self.sigmas_f:
            hphi_c_c.append(self.hx(s, **kwargs))
        self.hphi_c_c = np.atleast_2d(hphi_c_c)
        # recompute the mean and predictive measurement variance
        self.x_c_c, self.S = self.UT(
            sigmas = self.hphi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = np.zeros((self._dim_x, self._dim_x))
        )
        
    