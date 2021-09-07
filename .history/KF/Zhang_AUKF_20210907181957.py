import numpy as np
from numpy import dot
from copy import copy, deepcopy

class AdaptiveUnscentedKalmanFilter(object):
    '''
    Implementation of paper
    S. Zhang, “An Adaptive Unscented Kalman Filter for Dead Reckoning Systems,” in 2009 International Conference on Information Engineering and Computer Science, Dec. 2009, pp. 1–4, doi: 10.1109/ICIECS.2009.5365064.
    '''
    def __init__(
        self, 
        dim_z, 
        dim_x, 
        fx, 
        hx, 
        points, 
        Q0, 
        R0, 
        d,
        options = {'compute_log_likelihood' : True}, 
        FE_deriv_fn = None
        ):
        self._dim_z         =   dim_z
        self._dim_x         =   dim_x
        self.z              =   np.zeros(self._dim_z)
        self.fx             =   fx
        self.hx             =   hx
        self.points_fn      =   points
        self.Wm             =   points.Wm
        self.Wc             =   points.Wc
        self._num_sigmas    =   points.num_sigmas()
        self.Q              =   Q0
        self.R              =   R0
        self.d              =   d # forgetting scale
        self.sigmas_f       =   np.zeros((self._num_sigmas, self._dim_z))
        self.sigmas_h       =   np.zeros((self._num_sigmas, self._dim_x))
        self.options        =   options
        self.FE_deriv_fn    =   FE_deriv_fn
    
    @staticmethod
    def UT(sigmas, Wm, Wc, noise_cov):
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
        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = np.subtract(sigmas_h[i], x)
            dz = np.subtract(sigmas_f[i], z)
            Pxz += self.Wc[i] * np.outer(dz, dx)
        return Pxz

    def predict0(self, z0, P0, **kwargs):
        '''
        Note
        ----
        PRML uses initial guess of KF as pre-noise loaded prediction at time 1.
        This won't work in our case as well since we require the first guess to take
        into consideration the time varying parameter at time 1.
        We adapt it such that the first guess z0, P0 is the updated z and P at time -1.

        This may cause initial variations in analysis, however.
        A possible solution is to start analysing at time 2 instead of 1 to attenuate this effect.
        '''
        self.z = z0
        self.P = P0

        # We generate sigma points straight from the initial distribution
        sigmas = self.points_fn.sigma_points(self.z, self.P)
        for i,s in enumerate(sigmas):
            self.sigmas_f[i] = s
        
        self.z_og = self.z
        self.P_og = self.P
        
    def predict(self, **kwargs):
        # calculate sigma points for the given mean(z) and covariance(P)
        self.compute_process_sigmas(self.fx, **kwargs)
        # pass sigma points through unscented transform to compute P(Z_{t+1})
        self.z, self.P = AdaptiveUnscentedKalmanFilter.UT(
            sigmas = self.sigmas_f,
            Wm = self.Wm,
            Wc = self.Wc,
            noise_cov = self.Q
        )
        # update sigma points to reflect the new variance
        self.sigmas_f = self.points_fn.sigma_points(self.z, self.P)

        self.z_og = self.z
        self.P_og = self.P

    def update(self, x, **kwargs):
        # pass sigma points through hx to create measurement sigmas
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s, **kwargs))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        # mean and covariance of prediction passed through unscented transform
        self.xp, self.S = AdaptiveUnscentedKalmanFilter.UT(
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
        
        if self.options['compute_log_likelihood'] is True:
            self.compute_log_likelihood()

        self.z_updated = self.z
        self.P_updated = self.P

    def adaptive_update(self, x, i, **kwargs):
        '''
        As is evident, both Q and R adjustments are made using innovation.
        The formulation below makes it so that we can't sufficiently guarantee that this will be PD.
        Further adaption in future for this is required for stability.
        '''
        self.dk_1 = (1-self.d)/(1-self.d**(i))
        self.R = self.R - self.dk_1 * (np.outer(self.innovation, self.innovation) - self.S)
        # AUKF paper
        Q = self.Q + self.dk_1 * (
            np.outer(
                dot(self.K, self.innovation), 
                dot(self.K, self.innovation)
            ) + self.P_updated - self.P_og
        )
        input_dict = deepcopy(kwargs)
        kwargs.update({'Sigma'  :   self.R[0,0]})
        self.update(x, **kwargs)
        #self.Q = Q
    
    def compute_log_likelihood(self):
        self.mahalanobis = self.innovation.reshape(-1,1).T @ self.SI @ self.innovation.reshape(-1,1)
        self.log_likelihood = -1/2 * (np.log(np.linalg.det(self.S)) + 2 * np.log(2 * np.pi) + self.mahalanobis)
        
    