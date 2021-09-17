# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# As close of an exact implementation of DOI:10.3390/s18030808 (Zheng et al., Sensors, 2018).
# 2.
# Both Q,R are adapted. This derivation ensures PDness of resultant Q and R.
# 3.
# In-sync correction to state mean and variance occur. As such, the effect of the innovation
# and residual affect the filter immediately at the same time point.
# 4.
# Depending on the parameter, the algorithm acts as a fault detection in sequences.
# Threshold is only met when chi_sq_threshold value is breached.
# 5.
# This method belongs to a class of 'residual based adaptive method'.
# For more information on different types of adaptation possible, see DOI:10.1109/TAC.1972.1100100.
# For more information on derivation of 'residual based adaptive method', see DOI:10.1007/s001900050236.
# 6.
# Similar implementations of adaptive filters can be found in a variety of literature;
#   DOI: 10.3390/app9091726 (Battery Health)*
#   DOI: 10.1007/s001900050236 (GPS/INS)
#   *:A variation where hyperparameter is used in place describing noise as a distribution initself is also found


import numpy as np
from numpy import dot
from copy import copy, deepcopy

class AdaptiveUnscentedKalmanFilter:
    '''
    Adaptive Unscented Kalman Filter, as told by Zheng et al. (DOI:10.3390/s18030808).

    Parameters
    ----------
    kwargs  :   dict
        + n                 :   int
            # of iterations
        + chi_sq_threshold  :   float
            chi squared threshold value.
            Check your own data input (degree of freedom, confidence level, etc)
        + tune0             :   float
            Adaptation level baseline
        + a                 :   int
            Adaptation level

    ------------------------------------------------------------------------------------
    Naming convention
    ------------------------------------------------------------------------------------
    c              :   current
    p              :   past
    e.g., phi_c_p (predictive hidden state particle dependent on past)
    ------------------------------------------------------------------------------------
    Steps
    ------------------------------------------------------------------------------------
    1). psi computation (adaptation critera)
    psi         :   innovation.T * inv(Pxx_c_p) * innovation
        chi-squared statistic with a degree of freedom of the dimension of innovation vector
    2). Adaptive adjustment of Q
    tune        :   max(tune0, (psi - a * chi_sq_threshold)/psi)
        lambda
    Q           :   (1-lambda) * Q + lambda * (K * innovation * innovation.T * K)
        adjusted Q
    2). Adaptive adjustment of R
    residual    :   x - hx(z_c_c)
        residual vector between data and the predicted data from the updated z
    phi_c_c     :   points_fn.sigma_points(z,P)
        current updated sigma particles
    hphi_c_c    :   hx(phi_c_c)
        phi_c_c having gone through hx
    x_c_c       :   sum(w * hphi_c_c)
        mean of hphi_c_c
    S           :   sum(w[hphi_c_c - x_c_c][hphi_c_c - x_c_c])
        RAUKF process equivalent of Pxx_c_p (but without addition of R)
    R           :   (1-lambda) * R + lambda * (residual * residual.T + S)
    3). Correct estimates
    Pzz_c_c     :   sum(w[phi_c_c - z_c_c][phi_c_c - z_c_c])
        updated error covariance
    Pzx_c_c     :   sum(w[phi_c_c - z_c_c][hphi_c_c - x_c_c])
        updated cross covariance
    Pxx_c_c     :   sum(w[hphi_c_c - x_c_c][hphi_c_c - x_c_c]) + R
        updated innovation covariance (S + R)
    K           :   Pzx_c_c * inv(Pxx_c_c)
        updated Kalman gain
    z_c_c       :   UKF_updated z + updated Kalman gain * (data - updated predicted data)
                    z_c_c + K * (x - x_c_c)
        updated hidden state
    Pzz_c_c     :   UKF_updated P - updated Kalman gain * Pzx_c_c * updated Kalman gain.T
        updated error covariance
    We still adhere to the previous convention of naming phi as sigma_f and hphi as sigma_h.
    We, however, modify sigma_h such that it shows the time index such as sigma_h_c_p.
    '''
    def __init__(self, **kwargs):
        for attr in ['chi_sq_threshold', 'tune0', 'a']:
            if attr not in kwargs.keys():
                if attr == 'chi_sq_threshold':
                    self.chi_sq_threshold = 0.2110 # very wrong unless you change this yourself
                if attr == 'tune0':
                    self.tune0 = 0.1
                if attr == 'a':
                    self.a = 5
            else:
                setattr(self,attr,kwargs[attr])
        self.n              =   kwargs['n']
        self._adapted_idx   =   np.zeros((self.n,1),dtype=bool)
    
    def adapt_noise(self,x,**kwargs):
        # psi
        self.psi = dot(dot(self.innovation.reshape(1,-1), self.IPxx_c_p), self.innovation.reshape(-1,1))
        if self.psi > self.chi_sq_threshold:
            if 'i' in kwargs.keys():
                self._adapted_idx[kwargs['i']] = 1
                print('{}th iteration, psi threshold reached'.format(kwargs['i']))
            else:
                print('Psi threshold reached.')
            self.tune = max(self.tune0, (self.psi - self.a * self.chi_sq_threshold)/self.psi)
            self.adaptive_QR(x,**kwargs)
            self.correct_update(x)
    
    def adaptive_QR(self, x, **kwargs):
        self._adaptive_Q()
        self._adaptive_R(x,**kwargs)
    
    def _adaptive_Q(self):
        self.Q = (1-self.tune) * self.Q + self.tune * np.outer(
            dot(self.K, self.innovation), 
            dot(self.K, self.innovation)
        )
    
    def _adaptive_R(self,x,**kwargs):
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
    
    def correct_update(self,x):
        # error covariance (predictive variance)
        self.z_c_c, self.Pzz_c_c = self.UT(
            sigmas = self.phi_c_c,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        # cross covariance
        self.Pzx_c_c = self.cross_variance(
            z = self.z_c_c,
            x = self.x_c_c,
            sigmas_f = self.phi_c_c,
            sigmas_h = self.hphi_c_c
        )
        self.Pxx_c_c = self.S + self.R
        self.IPxx_c_c = np.linalg.inv(self.Pxx_c_c)
        self.K = dot(self.Pzx_c_c, self.IPxx_c_c)
        self.innovation = np.subtract(x,self.x_c_c)
        
        # correct-updated values
        self.z = self.z + dot(self.K, x - self.x_c_c)
        self.P = self.Pzz_c_c - dot(self.K, dot(self.Pxx_c_c, self.K.T))

        # Only done in order to compute likelihood
        self.S = self.Pxx_c_c


    def post_update(self):
        # distinction of being a posterior
        self.z_c_c = np.copy(self.z)
        self.Pzz_c_c = np.copy(self.P)
        self.compute_likelihood(self.innovation,self.S)