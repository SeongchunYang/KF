# Author: Seongchun Yang
# Affiliation: Kyoto University


# --------------------------------------------------------------------------------
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
    n           :   int
        number  of iterations for the filter
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
    def __init__(self, dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R, fading_memory = None, alpha_sq = None):
        self._dim_z         =   dim_z
        self._dim_x         =   dim_x
        self.z              =   z0 
        self.P              =   P0 
        self.Q              =   Q 
        self.R              =   R 
        self.fx             =   fx 
        self.hx             =   hx 
        self.points_fn      =   points_fn
        self.Wm             =   points_fn.Wm
        self.Wc             =   points_fn.Wc
        self._num_sigmas    =   points_fn.num_sigmas()
        self.sigmas_f       =   np.zeros((self._num_sigmas, self._dim_z))
        self.sigmas_h       =   np.zeros((self._num_sigmas, self._dim_x))

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
        self.z_og = self.z
        self.P_og = self.P
        
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
        self.z_og = self.z
        self.P_og = self.P

    def update(self, x, **kwargs):
        # pass sigma points through hx to create measurement sigmas
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s, **kwargs))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        # mean and covariance of prediction passed through unscented transform
        self.xp, self.S = self.UT(
            sigmas = self.sigmas_h,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.SI = np.linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        self.Pzx = self.cross_variance(
            x = self.xp,
            z = self.z,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h
        )
        self.K = dot(self.Pzx, self.SI)
        self.innovation = np.subtract(x, self.xp)

        # update
        self.z = self.z + dot(self.K, self.innovation)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

    def post_update(self):
        # distinction of being a posterior
        self.z_updated = np.copy(self.z)
        self.P_updated = np.copy(self.P)
        
        self.compute_log_likelihood()
    
    def compute_log_likelihood(self):
        self.mahalanobis = self.innovation.reshape(-1,1).T @ self.SI @ self.innovation.reshape(-1,1)
        self.log_likelihood = -1/2 * (np.log(np.linalg.det(self.S)) + 2 * np.log(2 * np.pi) + self.mahalanobis)