# Author: Seongchun Yang
# Affiliation: Kyoto University

import numpy as np
from numpy import dot
from copy import copy, deepcopy

class Schwarz_AUKF(object):
    '''
    Implementation of paper
    A. H. Mohamed and K. P. Schwarz, “Adaptive Kalman Filtering for INS/GPS,” Journal of Geodesy, vol. 73, no. 4, pp. 193–203, May 1999, doi: 10.1007/s001900050236.
    for UKF. Note that the authors never meant this to be used for anything other than a canonical Kalman filter.

    Q update for most algorithms present face non-Positive Definite issue due to the way it is computed, mainly that
    taking two PD matrices and subtracting them does not guarantee PD.
    '''
    def __init__(
        self, 
        n,
        dim_z, 
        dim_x,
        z0,
        P0,
        fx, 
        hx, 
        points_fn, 
        Q0, 
        R0, 
        L # Window length
        ):
        self._dim_z         =   dim_z
        self._dim_x         =   dim_x
        self.n              =   n
        self.z              =   np.zeros(self._dim_z)
        self.corrections    =   np.zeros((n,dim_z))
        self.corrections_v  =   np.zeros((n,dim_z,dim_z))
        self.residuals      =   np.zeros((n,dim_x))
        self.residuals_v    =   np.zeros((n,dim_x,dim_x))
        self.fx             =   fx
        self.hx             =   hx
        self.points_fn      =   points_fn
        self.Wm             =   points_fn.Wm
        self.Wc             =   points_fn.Wc
        self._num_sigmas    =   points_fn.num_sigmas()
        self.Q              =   Q0
        self.R              =   R0
        self.L              =   L 
        self.sigmas_f       =   np.zeros((self._num_sigmas, self._dim_z))
        self.sigmas_h       =   np.zeros((self._num_sigmas, self._dim_x))
    
    def UT(self,sigmas, Wm, Wc, noise_cov):
        kmax, n = sigmas.shape
        z = dot(Wm, sigmas)
        residual = sigmas - z[np.newaxis,:]
        P = dot(residual.T, dot(np.diag(Wc), residual)) + noise_cov
        return z, P

    def compute_process_sigmas(self, fx, z = None, P = None, **kwargs):
        if z is None:
            z = self.z
        if P is None:
            P = self.P
        # We generate sigma points from prescribed mean and covariance
        sigmas = self.points_fn.sigma_points(z, P)
        # Save sigma points (designated f to denote having been passed through fx)
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.fx(s, **kwargs)

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        # compute covariance between x (observation) and z (hideen states)
        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = np.subtract(sigmas_h[i], x)
            dz = np.subtract(sigmas_f[i], z)
            Pxz += self.Wc[i] * np.outer(dz, dx)
        return Pxz

    def predict0(self, **kwargs):
        # We generate sigma points straight from the initial distribution
        sigmas = self.points_fn.sigma_points(self.z, self.P)
        for i,s in enumerate(sigmas):
            self.sigmas_f[i] = s
        
        # distinction of being a prior
        self.z_og = self.z
        self.P_og = self.P
        
    def predict(self, **kwargs):# calculate sigma points for the given mean(z) and covariance(P)
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

        self.z_og = self.z
        self.P_og = self.P

    def update(self, x, i, **kwargs):
        '''
        The user has to manually input Sigma and hat_P
        '''
        # pass sigma points through hx to create measurement sigmas
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s, **kwargs))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        # mean and covariance of prediction passed through unscented transform
        self.xp, self.S = WindowUnscentedKalmanFilter.UT(
            sigmas = self.sigmas_h,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.SI = np.linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(
            x = self.xp,
            z = self.z_og,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h
        )
        self.K = dot(Pxz, self.SI)
        self.innovation = np.subtract(x, self.xp)

        # update - conventional KF
        self.z = self.z_og + dot(self.K, self.innovation)
        self.P = self.P_og - dot(self.K, dot(self.S, self.K.T))

        # update - FEP based mu update
        if self.FE_deriv_fn is not None:
            self.z[0]   =   self.z_og[0] - kwargs['kappa'] * self.FE_deriv_fn(
                vt          =   x[1],
                xt          =   x[0],
                Sigma       =   self.R[0,0],
                mu          =   self.z_og[0],
                hat_sigmat  =   self.P_og[0,0]
            )
        
        # update - corrections and residuals
        self.corrections[i,:] = np.subtract(self.z, self.z_og)
        self.corrections_v[i,:,:] = np.outer(
            np.subtract(self.z, self.z_og),
            np.subtract(self.z, self.z_og)
        )
        self.residuals[i,:] = np.subtract(x, self.hx(self.z, **kwargs))
        self.residuals_v[i,:,:] = np.outer(
            np.subtract(x, self.hx(self.z, **kwargs)),
            np.subtract(x, self.hx(self.z, **kwargs))
        )
        
        '''
        z_updated, P_updated designation will be given once it has gone through adaptive update.
        Hence, this class requires the user to use both at all times.
        '''

    def adaptive_update(self, x, i, **kwargs):
        '''
        The code takes care of adding right hat_P and Sigma.
        It loses generalization, however.
        '''
        if i < self.L:
            self._L = i
        else:
            self._L = L

        ### R
        ## Regenerate sigma points through updated z and P to calculate HP(k,+)H
        # -----------------------------------------------------------------------------
        # re-create sigma points corresponding to updated mean and varinace of hidden state
        self.phi_c_c = self.points_fn.sigma_points(self.z, self.P)
        # re-create sigma points corresponding to measurement sigma points
        hphi_c_c = []
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'hat_P' :   self.P,
            'Sigma' :   self.R[0,0]
        })
        for s in self.phi_c_c:
            hphi_c_c.append(self.hx(s, **input_dict))
        self.hphi_c_c = np.atleast_2d(hphi_c_c)
        # recompute the mean and predictive measurement variance
        _, self.S_c_c = WindowUnscentedKalmanFilter.UT(
            sigmas = self.hphi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = np.zeros((self._dim_x, self._dim_x))
        )
        # adapt R
        self.R = np.mean(
            self.residuals_v[i-self._L,:,:],
            axis = 0
        ) + self.S_c_c
        # -----------------------------------------------------------------------------

        ### Q
        # -----------------------------------------------------------------------------
        self.Q = np.mean(
            self.corrections_v[i-self._L,:,:],
            axis = 0
        ) + self.P - self.P_og
        # -----------------------------------------------------------------------------

        ### Correct estimates
        ## All steps in predict and update are repeated bearing in mind the new parameters
        # -----------------------------------------------------------------------------
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'hat_P' : self.P_updated,
            'Sigma' : self.R[0,0],
        })
        self.compute_process_sigmas(self.fx, z = self.z_updated, P = self.P_updated, **input_dict),
        self.z_corrected, self.P_corrected = WindowUnscentedKalmanFilter.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        self.sigmas_f = self.points_fn.sigma_points(self.z_corrected, self.P_corrected)
        
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'hat_P' : self.P_corrected,
            'Sigma' : self.R[0,0]
        })
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s,**input_dict))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        self.xp, self.S = WindowUnscentedKalmanFilter.UT(
            sigmas = self.sigmas_h,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.R
        )
        self.innovation = np.subtract(x, self.xp)
        self.SI = np.linalg.inv(self.S)
        self.Pzx = self.cross_variance(
            z = self.z_corrected,
            x = self.xp,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h
        )
        self.K = dot(self.Pzx, self.SI)

        # update final values
        self.z_updated = self.z_corrected + dot(self.K, np.subtract(x, self.xp))
        self.P_updated = self.P_corrected - dot(self.K, dot(self.S, self.K.T))

        self.corrections[i,:] = np.subtract(self.z_updated, self.z_corrected)
        self.corrections_v[i,:,:] = np.outer(
            np.subtract(self.z_updated, self.z_corrected),
            np.subtract(self.z_updated, self.z_corrected)
        )
        self.residuals[i,:] = np.subtract(x, self.hx(self.z_updated, **input_dict))
        self.residuals_v[i,:,:] = np.outer(
            np.subtract(x, self.hx(self.z_updated, **input_dict)),
            np.subtract(x, self.hx(self.z_updated, **input_dict))
        )

        if options['compute_log_likelihood'] is True:
            self.compute_log_likelihood()

    
    def compute_log_likelihood(self):
        self.mahalanobis = self.innovation.reshape(-1,1).T @ self.SI @ self.innovation.reshape(-1,1)
        self.log_likelihood = -1/2 * (np.log(np.linalg.det(self.S)) + 2 * np.log(2 * np.pi) + self.mahalanobis)