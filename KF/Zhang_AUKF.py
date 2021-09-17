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

class AdaptiveUnscentedKalmanFilter:
    '''
    Adaptive Unscented Kalman Filter, as told by Zhang et al. (DOI:10.1109/ICIECS.2009.5365064).
    
    Parameters
    ----------
    kwargs  :   dict
        + b     :   float (0<b<1)
            forgetting scale
    '''
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.b = kwargs['b']

    def adapt_noise(self, i, x, **kwargs):
        '''
        As is evident, both Q and R adjustments are made using innovation.
        The formulation below makes it so that we can't sufficiently guarantee that this will be PD.
        Further adaption in future for this is required for stability.
        '''
        self.d = (1-self.b)/(1-self.b**(i+2))
        self._adaptive_Q()
        self._adaptive_R()


    def _adaptive_R(self):
        self.R = self.R - self.d * (
            np.outer(
                self.innovation, 
                self.innovation
            ) - self.Pxx_c_p
        )

    def _adaptive_Q(self):
        self.Q = self.Q + self.d * (
            np.outer(
                dot(self.K, self.innovation),
                dot(self.K, self.innovation)
            ) + self.P - self.Pzz_c_p
        )
    
    def correct_update(self, x, **kwargs):
        '''
        (Optional)
        The paper doesn't necessarily elaborate on how one should integrate the updated Q and R.
        Here, a simple script which reuses the previously computed sigma points are presented here
        for reference. This allows the Q and R to update the current time estimates in sync.
        Note that the exact implementation may differ depending on the particulars of your filter
        and what the author expected. Here, UKF used allows the noise covariances to be added to the
        computed cross-covariance. Hence, simple replacement was deemd fit.
        '''
        # recompute mean and variance per updated Q
        self.z_c_c, self.Pzz_c_c = self.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        # recompute mean and variance per updated R
        self.x_c_c, self.Pxx_c_c = self.UT(
            sigmas = self.sigmas_h_c_p,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.IPxx_c_c = np.linalg.inv(self.Pxx_c_c)
        # recompute cross-covariance of the state and the measurements
        self.Pzx_c_c =self.cross_variance(
            z = self.z_c_c,
            x = self.x_c_c,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h_c_p
        )
        self.K = dot(self.Pzx_c_c, self.IPxx_c_c)
        self.innovation = np.subtract(x,self.x_c_c)
        # update
        self.z = self.z_c_c + dot(self.K, self.innovation)
        self.P = self.Pzz_c_c - dot(self.K, dot(self.Pxx_c_c, self.K.T))
        # purpose of computing likelihood
        self.S = self.Pxx_c_c

    def post_update(self, **kwargs):
        # distinction of being a posterior
        self.z_c_c = np.copy(self.z)
        self.Pzz_c_c = np.copy(self.P)    

        self.compute_log_likelihood(self.innovation,self.S)
        
    