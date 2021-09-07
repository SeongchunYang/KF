# Author: Seongchun Yang
# Affiliation: Kyoto University

import numpy as np
from numpy import dot
from copy import copy, deepcopy
from KF.UKF import UnscentedKalmanFilter

class adaptiveUKF(UnscentedKalmanFilter):
    '''
    Implementation of paper
    A. H. Mohamed and K. P. Schwarz, “Adaptive Kalman Filtering for INS/GPS,” Journal of Geodesy, vol. 73, no. 4, pp. 193–203, May 1999, doi: 10.1007/s001900050236.
    for UKF. Note that the authors never meant this to be used for anything other than a canonical Kalman filter.

    Features
    --------
    1. Q,R adaptation.
    For Q, it uses correction sequneces (z_updated - z_predicted). The formulation can lead to non-positive definite output,
    leading to instability in the filter where inverse does not exist. For R adaptation, residual sequences (x - x_predicted)
    are used. This leads to the formulation that guarantees the output to be PD.
    2. Integration of adjusted Q,R.
    Each iteration will first run the filter normally to find out the correction and residual.
    Then the filter will adjust Q and R. The adjusted Q and R are then used to re-compute the state,
    leading to instantaneous update to same time iteration.
    '''
    def __init__(
        self, 
        n,
        L, # Window length
        dim_z, 
        dim_x,
        z0,
        P0,
        fx, 
        hx, 
        points_fn, 
        Q, 
        R
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.n              =   n
        self.L              =   L 
        self.corrections    =   np.zeros((n,dim_z))
        self.corrections_v  =   np.zeros((n,dim_z,dim_z))
        self.residuals      =   np.zeros((n,dim_x))
        self.residuals_v    =   np.zeros((n,dim_x,dim_x))

    def adaptive_update(self, i, x, **kwargs):
        '''
        While prefix '_' is used for individual functions that are for Q and R, 
        the user is welcome to access these separately to implement only the 
        Q adaptation or the R adaptation.
        '''
        self._adaptive_stats(i,x,**kwargs)
        self._adaptive_updateQ(i)
        self._adaptive_updateR(i)
    
    def _adaptive_stats(self,i,x,**kwargs):
        if i < self.L:
            self._L = i
        else:
            self._L = L

        # corrections and residuals
        ## For Q
        self.corrections[i,:] = np.subtract(self.z, self.z_og)
        self.corrections_v[i,:,:] = np.outer(
            np.subtract(self.z, self.z_og),
            np.subtract(self.z, self.z_og)
        )
        ## For R
        self.residuals[i,:] = np.subtract(x, self.hx(self.z, **kwargs))
        self.residuals_v[i,:,:] = np.outer(
            np.subtract(x, self.hx(self.z, **kwargs)),
            np.subtract(x, self.hx(self.z, **kwargs))
        )

    def _adaptive_updateQ(self,i,**kwargs):
        self.Q = np.mean(
            self.corrections_v[i-self._L,:,:],
            axis = 0
        ) + self.P - self.P_og
    
    def _adaptive_updateR(self,i,**kwargs):
        '''
        Regenerate sigma points through updated z and P to calculate HP(k,+)H
        This is for the benefit of R adaptation only. Q does not need this portion.
        '''
        # re-create sigma points corresponding to updated mean and varinace of hidden state
        self.phi_c_c = self.points_fn.sigma_points(self.z, self.P)
        # re-create sigma points corresponding to measurement sigma points
        hphi_c_c = []
        for s in self.phi_c_c:
            hphi_c_c.append(self.hx(s, **kwargs))
        self.hphi_c_c = np.atleast_2d(hphi_c_c)
        # recompute the mean and predictive measurement variance
        _, self.S_c_c = self.UT(
            sigmas = self.hphi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = np.zeros((self._dim_x, self._dim_x))
        )
        self.R = np.mean(
            self.residuals_v[i-self._L,:,:],
            axis = 0
        ) + self.S_c_c
    
    def correct_update(self, x, **kwargs):
        '''
        If correct_update is applied to Q and R at the same time, Q must precede R in
        this step as the updated predicted state must be generated first.
        '''
        self._correct_updateQ(x,**kwargs)
        self._correct_updateR(x,**kwargs)
    
    def _correct_updateQ(self, x, **kwargs):
        '''
        Correct estimates using the updated Q.
        '''
        self.compute_process_sigmas(self.fx, **kwargs),
        self.z_corrected, self.P_corrected = self.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        self.sigmas_f = self.points_fn.sigma_points(self.z_corrected, self.P_corrected)
        
        # commmon util function for both Q and R
        self.__correct_updateQR(x,self.z_corrected,self.P_corrected)
    
    def _correct_updateR(self, x, **kwargs):
        '''
        Correct estimates using the updated R.
        '''
        self.__correct_updateQR(x,self.z,self.P)
    
    def __correct_updateQR(self,x,z_corrected,P_corrected,**kwargs):
        '''
        Common function for _correct_updateQ and R.
        Creates sigma points for observation and obtains the posterior.
        If _correct_updateQ was never ran, sigmas_f remain the same and can be
        used for _correct_updateR with just R adjustments.
        If _correct_updateQ was ran, sigmas_f will be recalculated from the  updated Q
        and will be used with either adjusted R or non-adjusted R depending on what is
        available through self.R.
        Parameters
        ----------
        x           :   array_like
            observation
        z_corrected :   array_like
            corrected prediction of state    
        '''
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s,**kwargs))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        self.xp, self.S = self.UT(
            sigmas = self.sigmas_h,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.innovation = np.subtract(x, self.xp)
        self.SI = np.linalg.inv(self.S)
        self.Pzx = self.cross_variance(
            z = z_corrected,
            x = self.xp,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h
        )
        self.K = dot(self.Pzx, self.SI)

        # update final values
        self.z = z_corrected + dot(self.K, np.subtract(x, self.xp))
        self.P = self.P_corrected - dot(self.K, dot(self.S, self.K.T))

        self.post_update() # update z_updated, P_updated and compute stats

        '''
        # if comparing corrections and residuals with the pre-adjustment, use this
        self._corrections[i,:] = np.subtract(self.z_updated, self.z_corrected)
        self._corrections_v[i,:,:] = np.outer(
            np.subtract(self.z_updated, self.z_corrected),
            np.subtract(self.z_updated, self.z_corrected)
        )
        self._residuals[i,:] = np.subtract(x, self.hx(self.z_updated, **kwargs))
        self._residuals_v[i,:,:] = np.outer(
            np.subtract(x, self.hx(self.z_updated, **kwargs)),
            np.subtract(x, self.hx(self.z_updated, **kwargs))
        )
        '''