# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANTL:CITATION ALERT)
# The MIT License (MIT)
# Copyright (c) 2015 Roger R. Labbe Jr
    # This script is largely an adaptation of the above copyright holder.
    # The original script can be found in https://github.com/rlabbe/filterpy
    # Differences between this and the original are as follows.
        # Treatment of variable names. 'z' is the state and 'x' is the observation.
        # Addition of initialization prediction step. To learn more, read the function description below.
        # Added modularity for use with other scripts in the repo.
# 2.
# Should the user wish to add more performance statistic measures, simply add some to 'compute_likelihood'
# function and extract at every iteration for use later.



import numpy as np
from numpy import dot

class UnscentedKalmanFilter(object):
    '''
    Base UKF script. For a typical run of UKF, one should do the following;
    while t < T:
        if t == 1:
            self.predict0(**kwargs)
            self.update(i,x,**kwargs)
            self.post_update(i,x,**kwargs)
        else:
            self.predict(**kwargs)
            self.update(i,x,**kwargs)
            self.post_update(i,x,**kwargs)
    Parameters
    ----------
    dim_z       :   int
        dimension of state vector
    dim_x       :   int
        dimension of observation vector
    z0          :   array_like
        initial state value
        has to be an array of size 1 (i.e., np.array([1]).shape == (1,))
    P0          :   array_like
        initial state variance, with 2d shape (i.e., for dim_z = 1, np.atleast_2d(1))
        has to be a 2D array
    fx          :   function
        state transition function
        i.e., def fx(sigma_point, **kwargs)
    hx          :   function
        observation function
        i.e., def hx(sigma_point, **kwargs)
    points_fn   :   function
        sigma point generator either suggested by Julier et al. or Merwe et al.
    '''
    def __init__(self, dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R):
        self._dim_z         =   dim_z
        self._dim_x         =   dim_x
        self.z              =   z0 
        self.P              =   P0 
        self.fx             =   fx 
        self.hx             =   hx 
        self.points_fn      =   points_fn
        self.Wm             =   points_fn.Wm
        self.Wc             =   points_fn.Wc
        self._num_sigmas    =   points_fn.num_sigmas()
        self.sigmas_f       =   np.zeros((self._num_sigmas, self._dim_z))
        self.sigmas_h       =   np.zeros((self._num_sigmas, self._dim_x))
        self.Q              =   Q 
        self.R              =   R

    def UT(self, sigmas, Wm, Wc, noise_cov):
        kmax, n = sigmas.shape
        z = dot(Wm, sigmas)
        residual = sigmas - z[np.newaxis,:]
        P = dot(residual.T, dot(np.diag(Wc), residual)) + noise_cov
        return z, P

    def compute_process_sigmas(self, fx, **kwargs):
        # We generate sigma points from prescribed mean and covariance
        sigmas = self.points_fn.sigma_points(self.z, self.P)
        # Save sigma points (designated f to denote having been passed through fx)
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.fx(s, **kwargs)

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        # compute covariance between x (observation) and z (hideen states)
        Pzx = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dz = np.subtract(sigmas_f[i], z)
            dx = np.subtract(sigmas_h[i], x)
            Pzx += self.Wc[i] * np.outer(dz, dx)
        return Pzx

    def predict0(self, **kwargs):
        '''
        Optional step at initialization.

        Case [1]
        In certain implementation of Kalman filters, the initial mean and variances are
        considered as previous time updated state. Then they are used to predict the current
        time step.
        Case [2]
        If not case [1], they may be used as the current time prediction for the first 
        time point state.
        
        This function is to be used for case [2]. Mean and variance inputs are intialized
        to create the sigma points which are considered a prediction of the current time.
        
        Example Syntax for case[1]
        --------------------------
        for i in range(n):
            UKF.predict(**kwargs)
            UKF.update(x,**kwargs)

        Example Syntax for case[2]
        --------------------------
        for i in range(n):
            if i == 0:
                UKF.predict0(z0,P0,**kwargs)
                UKF.update(x,**kwargs)
            else:
                UKF.predict(**kwargs) # z0,P0 should be preset
                UKF.update(x,**kwargs)
        '''
        # We generate sigma points straight from the initial distribution
        sigmas = self.points_fn.sigma_points(self.z, self.P)
        for i,s in enumerate(sigmas):
            self.sigmas_f[i] = s
        
        # distinction of being a prior
        self.z_c_p = self.z
        self.Pzz_c_p = self.P
        
    def predict(self, **kwargs):
        '''
        For detailed comparison with predict0 function, see def predict0.
        '''
        # calculate sigma points for the given mean(z) and covariance(P)
        self.compute_process_sigmas(self.fx, **kwargs)
        # pass sigma points through unscented transform to compute P(Z_{t+1})
        self.z, self.P = self.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        # update sigma points to reflect the new variance
        self.sigmas_f = self.points_fn.sigma_points(self.z, self.P)

        # distinction of being a prior
        self.z_c_p = self.z
        self.Pzz_c_p = self.P

    def update(self, x, **kwargs):
        '''
        Naming convention
        c              :   current
        p              :   past
        e.g., phi_c_p (predictive hidden state particle dependent on past)
        
        By progression... (At time t, given Pzz_p_p and z_p_p)
        UKF
        ------------------------------------------------------------------------------------

        1). Create sigma particles corresponding to pre-computed past mean and covariance
        phi_p_p     :   points_fn.sigma_points(z,P)
            past sigma particles
        2). Predict current state and its error covariance
        phi_c_p     :   compute_process_sigmas(fx, z, P)
            predictive sigma particles
        z_c_p       :   sum(w * phi_c_p)
            mean of predictive sigma particles
        Pzz_c_p     :   sum(w[phi_c_p - z_c_p][phi_c_p - z_c_p].T)
            error covariance matrix (variance of predictive marginal)
        3). Determine predicted innovation covariance matrix and cross-covariance matrix
        hphi_c_p    :   hx(phi_c_p)
            phi_c_p having gone through obsevation model hx
        innovation  :   x - hx(z_c_p)
            innovation vector
        x_c_p       :   sum(w * hphi_c_p)
            mean of hphi_c_p
        Pxx_c_p     :   sum(w[hphi_c_p - x_c_p][hphi_c_p - x_c_p].T) + R
            innovation covariance (variance of predictive likelihood marginal)
        Pzx_c_p     :   sum(w[phi_c_p - z_c_p][hphi_c_p - x_c_p].T)
            cross-covariance (covariance of likelihood and state transition)
        4). Calculate Kalman gain and obtain updated current estimates of state and its error covariance matrix
        K           :   Pzx_c_p * inv(Pxx_c_p)
            Kalman gain
        z           :   z_c_p + K * (x - x_c_p)
            updated state
        P           :   Pzz_c_p - K * Pxx_c_p * K.T
            updated state covariance
        '''
        # pass sigma points through hx to create measurement sigmas
        # ---------------------------------------------------------
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s, **kwargs))
        self.sigmas_h_c_p = np.atleast_2d(sigmas_h)
        
        # calculate post predictive mean and variance from sigma points via UT
        # ---------------------------------------------------------
        self.x_c_p, self.Pxx_c_p = self.UT(
            sigmas = self.sigmas_h_c_p,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.IPxx_c_p = np.linalg.inv(self.Pxx_c_p)

        # compute cross variance of the state and the measurements
        # ---------------------------------------------------------
        self.Pzx_c_p = self.cross_variance(
            z = self.z_c_p,
            x = self.x_c_p,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h_c_p
        )
        self.K = dot(self.Pzx_c_p, self.IPxx_c_p)
        self.innovation = np.subtract(x, self.x_c_p)

        # conventional-UKF-update
        # ---------------------------------------------------------
        self.z = self.z + dot(self.K, x - self.x_c_p)
        self.P = self.P - dot(self.K, dot(self.Pxx_c_p, self.K.T))

        self.compute_log_likelihood(self.innovation,self.Pxx_c_p)

    def post_update(self):
        # distinction of being a posterior
        self.z_c_c = np.copy(self.z)
        self.Pzz_c_c = np.copy(self.P)    
    
    def compute_log_likelihood(self,innovation,S):
        self.mahalanobis = innovation.reshape(-1,1).T @ np.linalg.inv(S) @ innovation.reshape(-1,1)
        self.log_likelihood = -1/2 * (np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi) + self.mahalanobis)