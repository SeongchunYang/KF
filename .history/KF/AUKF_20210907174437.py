# Author: Seongchun Yang
# Affiliation: Kyoto University


import numpy as np
from copy import copy
from copy import deepcopy
from KF.UKF import UnscentedKalmanFilter

class adaptiveUKF(UnscentedKalmanFilter):
    def __init__(
        self,
        n,
        delta,
        dim_z,
        dim_x,
        z0,
        P0,
        fx, 
        hx, 
        points_fn,
        Q, 
        R, 
        fading_memory = None,
        alpha_sq = None
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R, fading_memory, alpha_sq)
        self.n                  =   n
        self.delta              =   delta
        self.residuals          =   np.empty((n,dim_x))
        self.residual_variances =   np.empty((n,dim_x,dim_x))
    
    def adapt_R(self, i, x, **kwargs):
        '''
        Post-hoc adaptive measurement noise.
        Computes residuals which are used to construct a guaranteed positive definite matrix.
        Parameters
        ----------
        i   :   int
            iteration index
        x   :   array_like
            observation
        '''
        # re-create sigma points corresponding to updated mean and variance of hidden state
        self.phi_c_c = self.points_fn.sigma_points(self.z, self.P)
        
        # re-create sigma points corresponding to measurement sigma points
        hphi_c_c = []
        for s in self.phi_c_c:
            hphi_c_c.append(self.hx(s, **kwargs))
        self.hphi_c_c = np.atleast_2d(hphi_c_c)
        
        # recompute the mean and predictive measurement variance
        self.x_c_c, self.S_c_c = self.UT(
            sigmas = self.hphi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = np.zeros((self._dim_x, self._dim_x))
        )
        self.residual = np.subtract(x, self.x_c_c)
        self.residual_variance = np.outer(self.residual, self.residual)
        # save residual sequences
        self.residuals[i,:] = self.residual
        self.residual_variances[i,:,:] = self.residual_variance

        # R adaptation
        self.R = (1 - self.delta) * self.R + self.delta * (self.residual_variance + self.S_c_c)