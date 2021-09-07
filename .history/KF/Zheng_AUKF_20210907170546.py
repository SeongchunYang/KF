import numpy as np
from numpy import dot
from copy import copy, deepcopy
from KF.UKF import UnscentedKalmanFilter

class adaptiveUKF(UnscentedKalmanFilter):
    def __init__(self, dim_z, dim_x, zo, P0, fx, hx, points_fn, Q, R, chi_sq_threshold = None, tune0 = None, a = None):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.chi_sq_threshold = chi_sq_threshold or 0.2110
        self.tune0 = tune0 or 0.1
        self.a = a or 5
    
    def adapt_QR(self):
        # psi
        self.psi = dot(dot(self.innovation_vector.T, self.IPxx_c_p), self.innovation_vector)
        if self.psi > self.chi_sq_threshold:
            print('{}th iteration, psi threshold reached'.format(kwargs['i']))
            self.tune = max(self.tune0, (self.psi - self.a * self.chi_sq_threshold)/self.psi)
    
    def _adaptive_QR(self, x, **kwargs):
        
        # ----------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------
        # Q update
        # -----------------------------------------------------
        self.tune = self.tune0
        self.innovation = np.subtract(x, self.x_c_p)
        self.Q = (1-self.tune) * self.Q + self.tune * np.outer(
            dot(self.K, self.innovation), 
            dot(self.K, self.innovation)
        )
            
        # R update
        # -----------------------------------------------------
        # re-create sigma points corresponding to updated mean and variance of hidden state
        self.phi_c_c = self.points_fn.sigma_points(self.z, self.P)
        # re-create sigma points corresponding to measurement sigma points
        hphi_c_c = []
        for s in self.phi_c_c:
            hphi_c_c.append(self.hx(s, **kwargs))
        self.hphi_c_c = np.atleast_2d(hphi_c_c)
        # recompute the mean and predictive measurement variance
        self.x_c_c, self.S = self.UT(
            sigmas = self.hphi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = np.zeros((self._dim_x, self._dim_x))
        )
        #self.residual = np.subtract(x, self.hx(self.z, **kwargs))
        self.residual = np.subtract(x, self.x_c_c)
        self.R = (1-self.tune) * self.R + self.tune * (
            np.outer(
                self.residual, 
                self.residual
            ) + self.S
        )
    
    def 
        '''
        A distinctive feature here (which is important in connection to the theory of covariance matching in
        adaptve noise estimation) is that Q and R are being estimated independently
        (as in, it doesn't matter which comes first)
        '''

        # ----------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------
        # correct estimates
        # -----------------------------------------------------
        # error covariance (predictive variance)
        self.z_c_c, self.Pzz_c_c = self.UT(
            sigmas = self.phi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        if np.allclose(self.z_c_c, self.z) is False:
            print(self.z_c_c, self.z)
        # cross covariance
        self.Pzx_c_c = self.cross_variance(
            z = self.z_c_c,
            x = self.x_c_c,
            sigmas_f = self.phi_c_c,
            sigmas_h = self.hphi_c_c
        )
        self.Pxx_c_c = self.S + self.R
        self.IPxx_c_c = self.inv(self.Pxx_c_c)
        self.K = dot(self.Pzx_c_c, self.IPxx_c_c)
        
        # updated values
        self.z = self.z + dot(self.K, x - self.x_c_c)
        #self.P = self.Pzz_c_c - dot(self.K, dot(self.Pzx_c_c, self.K.T))
        #self.P = self.Pzz_c_c - self.K @ self.Pxx_c_c @ self.K.T
        self.P = self.Pzz_c_c - dot(self.K, dot(self.Pxx_c_c, self.K.T))
        
        self.z_c_c = deepcopy(self.z)
        self.Pzz_c_c = deepcopy(self.P)
