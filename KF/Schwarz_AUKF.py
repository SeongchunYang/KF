# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# Implementation of paper doi: 10.1007/s001900050236 by Mohamed and Schwarz.
# Note that the authors never meant this to be used for anything other than a canonical Kalman filter.
# As such, application to UKF was done by the author of this code.
# 2. 
# Both Q,R adaptations are done. For Q, it uses correction sequneces (z_updated - z_predicted). 
# The formulation can lead to non-positive definite output, leading to instability in the filter 
# where inverse does not exist. For R adaptation, residual sequences (x - x_predicted)
# are used. This leads to the formulation that guarantees the output to be PD.
# 3.
# Each iteration will first run the filter normally to find out the correction and residual.
# Then the filter will adjust Q and R. The adjusted Q and R are then used to re-compute the 
# state, leading to instantaneous update to same time iteration.
# 4.
# This method belongs to a class of 'residual based adaptive method'.
# For more information on different types of adaptation possible, see DOI:10.1109/TAC.1972.1100100.
# For more information on derivation of 'residual based adaptive method', see DOI:10.1007/s001900050236.
# 5.
# Similar implementations of adaptive filters can be found in a variety of literature;
#   DOI: 10.3390/app9091726 (Battery Health)*
#   DOI: 10.1007/s001900050236 (GPS/INS)
#   *:A variation where hyperparameter is used in place describing noise as a distribution initself is also found

# NOTE
# This script is currently highly unstable due to the way Q is adapted. Do not use
# unless adapted for your own needs.

import numpy as np
from numpy import dot
from copy import copy, deepcopy

class adaptiveUKF:
    '''
    A mixin component.
    This is not meant as a standalone filter.
    In order to utilize this as a full adaptive filter, do the following;
        Example (with UKF):
        class constructor_AUKF(UnscentedKalmanFilter,AdaptiveUnscentedKalmanFilter,metaclass=MixedClassMeta):
            def __init__(self,*args,**kwargs): pass

        AUKF = constructor_AUKF(**kwargs)
    
    Parameters
    ----------
    kwargs  :   dict
        + n    :    # of iterations
        + L    :    window length
    '''
    def __init__(self, **kwargs):
        self.n              =   kwargs['n']
        self.L              =   kwargs['L']
        self.corrections    =   np.empty((self.n,self._dim_z))
        self.corrections_v  =   np.empty((self.n,self._dim_z,self._dim_z))
        self.residuals      =   np.empty((self.n,self._dim_x))
        self.residuals_v    =   np.empty((self.n,self._dim_x,self._dim_x))

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
        self.P = P_corrected - dot(self.K, dot(self.S, self.K.T))
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