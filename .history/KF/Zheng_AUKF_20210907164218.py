import numpy as np
from numpy import dot
from copy import copy, deepcopy
from KF.UKF import UnscentedKalmanFilter

class adaptiveUKF:
    def __init__(self, dim_z, dim_x, fx, hx, points_fn, Q, R, chi_sq_threshold = None, tune0 = None, a = None, inv = None):
        #super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self._dim_z = dim_z
        self._dim_x = dim_x
        self.Q = Q
        self.R = R
        self.z = np.zeros(self._dim_z)
        self.fx = fx
        self.hx = hx
        self.points_fn = points
        self.Wm = points.Wm
        self.Wc = points.Wc
        self._num_sigmas = points.num_sigmas()
        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_z))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_x))
        self.chi_sq_threshold = chi_sq_threshold or 0.2110
        self.tune0 = tune0 or 0.1
        self.a = a or 5
        self.inv = inv or np.linalg.inv
    
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

    def predict0(self, z0, P0, **kwargs):
        self.z = z0
        self.P = P0

        # We generate sigma points straight from the initial distribution
        sigmas = self.points_fn.sigma_points(self.z, self.P)
        for i,s in enumerate(sigmas):
            self.sigmas_f[i] = s
        
        self.z_c_p = self.z
        self.Pzz_c_p = self.P
        
    def predict(self, **kwargs):
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
        self.IPxx_c_p = self.inv(self.Pxx_c_p)

        # compute cross variance of the state and the measurements
        # ---------------------------------------------------------
        self.Pzx_c_p = self.cross_variance(
            z = self.z_c_p,
            x = self.x_c_p,
            sigmas_f = self.sigmas_f,
            sigmas_h = self.sigmas_h_c_p
        )
        self.K = dot(self.Pzx_c_p, self.IPxx_c_p)

        # conventional-UKF-update
        # ---------------------------------------------------------
        self.z = self.z + dot(self.K, x - self.x_c_p)
        self.P = self.P - dot(self.K, dot(self.Pxx_c_p, self.K.T))
    
        '''
        # psi
        self.psi = dot(dot(self.innovation_vector.T, self.IPxx_c_p), self.innovation_vector)
        if self.psi > self.chi_sq_threshold and self.mode == 'robust':
            print('{}th iteration, psi threshold reached'.format(kwargs['i']))
            self.tune = max(self.tune0, (self.psi - self.a * self.chi_sq_threshold)/self.psi)
        '''
        self.compute_likelihood(x, self.x_c_p, self.Pxx_c_p)
        
        self.z_c_c = deepcopy(self.z)
        self.Pzz_c_c = deepcopy(self.P)
    
    def adaptive_Q_R(self, x, **kwargs):
        '''
        Naming convention
        c              :   current
        p              :   past
        e.g., phi_c_p (predictive hidden state particle dependent on past)

        RAUKF
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
        # ----------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------

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
    
    def compute_likelihood(self, x, xp, Sp):
        self.log_likelihood = scipy.stats.multivariate_normal.logpdf(
            x = x,
            mean = xp,
            cov = Sp,
            allow_singular = True
        )
