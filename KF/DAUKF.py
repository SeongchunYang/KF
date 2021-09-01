import numpy as np
from copy import copy
from copy import deepcopy
from FEPpkg.filters.UKF import UnscentedKalmanFilter

class adaptiveUKF(UnscentedKalmanFilter):
    def __init__(
        self, 
        n, 
        delta, 
        dim_z, 
        dim_x, 
        fx, 
        hx, 
        points,
        Q, 
        R, 
        options = None, 
        FE_deriv_fn = None,
        multi_FE_deriv_fn = None
        ):
        super().__init__(dim_z, dim_x, fx, hx, points, Q, R, options, FE_deriv_fn, multi_FE_deriv_fn)
        self.n                  =   n
        self.delta              =   delta
        self.residuals          =   np.zeros((n,dim_x))
        self.residual_variances =   np.zeros((n,dim_x,dim_x))
    
    def adapt_R(self, i, x, **kwargs):
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
    
    def _adapt_R(self, i, x, **kwargs):
        self.R[0,0] = (1 - self.delta) * self.R[0,0] + self.delta * self.innovation[0]**2

class stateUKF(adaptiveUKF):
    def __init__(
        self, 
        n, 
        delta, 
        dim_z, 
        dim_x, 
        fx, 
        hx, 
        points, 
        Q, 
        R, 
        options = None, 
        FE_deriv_fn = None, 
        multi_FE_deriv_fn = None
        ):
        super().__init__(n, delta, dim_z, dim_x, fx, hx, points, Q, R, options, FE_deriv_fn, multi_FE_deriv_fn)
        self.zs     =   np.zeros((n,dim_z))
        self.xps    =   np.zeros((n,dim_x))
        self.Pzxs   =   np.zeros((n,dim_z,dim_x))
        self.Ps     =   np.zeros((n,dim_z,dim_z))
        self.Ss     =   np.zeros((n,dim_x,dim_x))
        self.Qs     =   np.zeros((n,dim_z,dim_z))
        self.Rs     =   np.zeros((n,dim_x,dim_x))

        # stats
        self.innovations     =  np.zeros((n,dim_x))
        self.mahalanobiss    =  np.zeros((n,1))
        self.log_likelihoods =  np.zeros((n,1))

    def initial_parameters(self, x1, v1, c0, **kwargs):
        # params
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'R'         :   self.R,
            'hat_P'     :   self.P_updated + self.Q,
            'c'         :   c0
        })
        try:
            ## predict
            # ---------------------------
            self.predict(**input_dict)
            
            ## update
            # ---------------------------
            self.update(
                x = (x1,v1),
                **input_dict
            )

            # no adaptR called for initial parametrization
        except:
            print('Initial parameter adjustment in state filter failed.')
            raise

    def iteration(self, i, x, v, c, **kwargs):
        # params
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'R'         :   self.R,
            'hat_P'     :   self.P_updated + self.Q,
            'c'         :   c
        })
        try:
            ## predict
            # ---------------------------
            self.predict(**input_dict)
            
            ## update
            # ---------------------------
            self.update(
                x = (x,v),
                **input_dict
            )
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_updated
            self.xps[i,:]    =   self.xp
            self.Pzxs[i,:,:] =   self.Pzx
            self.Ps[i,:,:]   =   self.P_updated
            self.Ss[i,:,:]   =   self.S
            self.Qs[i,:,:]   =   self.Q
            self.Rs[i,:,:]   =   self.R
            # stats
            self.innovations[i,:]       =   self.innovation.ravel()
            self.mahalanobiss[i,:]      =   self.mahalanobis
            self.log_likelihoods[i,:]   =   self.log_likelihood

            ## Adapt R
            # ---------------------------
            if self.options['adapt_R']:
                self.adapt_R(i, (x,v))
        
        except:
            print('{}th iteration in state filter threw an error.'.format(i))
            raise

class parameterUKF(adaptiveUKF):
    def __init__(
        self, 
        n, 
        delta, 
        dim_z, 
        dim_x, 
        fx, 
        hx, 
        points, 
        Q,
        R, 
        options = None
        ):
        super().__init__(n, delta, dim_z, dim_x, fx, hx, points, Q, R, options)
        self.zs     =   np.zeros((n,dim_z))
        self.xps    =   np.zeros((n,dim_x))
        self.Pzxs   =   np.zeros((n,dim_z,dim_x))
        self.Ps     =   np.zeros((n,dim_z,dim_z))
        self.Ss     =   np.zeros((n,dim_x,dim_x))
        self.Qs     =   np.zeros((n,dim_z,dim_z))
        self.Rs     =   np.zeros((n,dim_x,dim_x))

        # stats
        self.innovations     =  np.zeros((n,dim_x))
        self.mahalanobiss    =  np.zeros((n,1))
        self.log_likelihoods =  np.zeros((n,1))
    
    def initial_parameters(self, x1, v1, z1, hat_P1, **kwargs):
        # params
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'R'         :   self.R,
            'z'         :   z1,
            'hat_P'     :   hat_P1
        })
        try:
            ## predict
            # ---------------------------
            self.predict()
            
            ## update
            # ---------------------------
            self.update(
                x = (x1,v1),
                **input_dict
            )

            # no adaptR called for initial parametrization
        except:
            print('Initial parameter adjustment in state filter failed.')
            raise

    def iteration(self, i, x, v, state_z, state_hat_P, **kwargs):
        # params
        input_dict = deepcopy(kwargs)
        input_dict.update({
            'R'         :   self.R, 
            'z'         :   state_z,
            'hat_P'     :   state_hat_P
        })
        try:
            ## predict
            # ---------------------------
            self.predict()
            
            ## update
            # ---------------------------
            self.update(
                x = (x,v),
                **input_dict
            )
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_updated
            self.xps[i,:]    =   self.xp
            self.Pzxs[i,:,:] =   self.Pzx
            self.Ps[i,:,:]   =   self.P_updated
            self.Ss[i,:,:]   =   self.S
            self.Qs[i,:,:]   =   self.Q
            self.Rs[i,:,:]   =   self.R
            # stats
            self.innovations[i,:]       =   self.innovation.ravel()
            self.mahalanobiss[i,:]      =   self.mahalanobis
            self.log_likelihoods[i,:]   =   self.log_likelihood

            ## Adapt R
            # ---------------------------
            if self.options['adapt_R']:
                self.adapt_R(
                    i,
                    (x,v),
                    **input_dict
                )
            
        except:
            print('{}th iteration in parameter filter threw an error.'.format(i))
            raise

class DualAdaptiveUKF(object):
    '''
    The primary reason we are employing this nested class is due to having to initiate two separate class objects
    of the same parent that are run concurrently.
    '''
    def __init__(
        self, 
        data,
        delta,
        sUKF_args, 
        pUKF_args,
        s_Q,
        s_R,
        p_Q,
        p_R,
        n = None,
        FE_deriv_fn = None, 
        multi_FE_deriv_fn = None
        ):
        # state filter parameters
        self.s_z0               =   np.copy(sUKF_args['z0'])
        self.s_P0               =   np.copy(sUKF_args['P0'])
        self.s_Q                =   np.copy(s_Q)
        self.s_R                =   np.copy(s_R)
        self.s_dimz             =   sUKF_args['dim_z'] 
        self.s_dimx             =   sUKF_args['dim_x']
        self.s_fx               =   sUKF_args['fx']
        self.s_hx               =   sUKF_args['hx']
        self.s_points           =   sUKF_args['points_fn']
        self.s_options          =   sUKF_args['options']
        # parameter filter parameters
        self.p_z0               =   np.copy(pUKF_args['z0'])
        self.p_P0               =   np.copy(pUKF_args['P0'])
        self.p_Q                =   np.copy(p_Q)
        self.p_R                =   np.copy(p_R)
        self.p_dimz             =   pUKF_args['dim_z'] 
        self.p_dimx             =   pUKF_args['dim_x']
        self.p_fx               =   pUKF_args['fx']
        self.p_hx               =   pUKF_args['hx']
        self.p_points           =   pUKF_args['points_fn']
        self.p_options          =   pUKF_args['options']
        # others
        self.n                  =   n or data.tv_Speed.shape[0]
        self.delta              =   delta
        self.data               =   data
        self.FE_deriv_fn        =   FE_deriv_fn
        self.multi_FE_deriv_fn  =   multi_FE_deriv_fn
    
    def initiate(self):
        # Force clean slate
        if hasattr(self, 'sUKF'):
            delattr(self, 'sUKF')
        if hasattr(self, 'pUKF'):
            delattr(self, 'pUKF')
        
        # Initiate - State Filter
        self.sUKF = stateUKF(
            n                   =   self.n,
            delta               =   self.delta,
            dim_z               =   self.s_dimz,
            dim_x               =   self.s_dimx,
            Q                   =   self.s_Q,
            fx                  =   self.s_fx,
            hx                  =   self.s_hx,
            points              =   self.s_points,
            R                   =   self.s_R,
            options             =   self.s_options,
            FE_deriv_fn         =   self.FE_deriv_fn,
            multi_FE_deriv_fn   =   self.multi_FE_deriv_fn
        )
        self.sUKF.z = self.sUKF.z_updated = self.s_z0
        self.sUKF.P = self.sUKF.P_updated = self.s_P0

        # Initiate - Parameter Filter
        self.pUKF = parameterUKF(
            n                   =   self.n,
            delta               =   self.delta,
            dim_z               =   self.p_dimz,
            dim_x               =   self.p_dimx,
            Q                   =   self.p_Q,
            fx                  =   self.p_fx,
            hx                  =   self.p_hx,
            points              =   self.p_points,
            R                   =   self.p_R,
            options             =   self.p_options
        )
        self.pUKF.z = self.pUKF.z_updated = self.p_z0
        self.pUKF.P = self.pUKF.P_updated = self.p_P0
    
    def reparameterization(self, **kwargs):
        '''
        Find the optimal initialization point for the filter.
        Only use after initiation.
        '''
        #(previous)
        ps_z_updated = np.copy(self.sUKF.z_updated)
        ps_P_updated = np.copy(self.sUKF.P_updated)
        ps_Q = np.copy(self.sUKF.Q)

        # Find the optimal starting point
        self.sUKF.initial_parameters(
            x1 = self.data.observed_flows.ravel()[0],
            v1 = self.data.observed_actions.ravel()[0],
            c0 = self.pUKF.z,
            **kwargs
        )
        self.pUKF.initial_parameters(
            x1 = self.data.observed_flows.ravel()[0],
            v1 = self.data.observed_actions.ravel()[0],
            z1 = ps_z_updated,
            hat_P1 = ps_P_updated + ps_Q,
            **kwargs
        )

    def main(self, **kwargs):
        for i,(x,v) in enumerate(
            zip(self.data.observed_flows.ravel(), self.data.observed_actions.ravel())
            ):
            #(previous)
            ps_z_updated = np.copy(self.sUKF.z_updated)
            ps_P_updated = np.copy(self.sUKF.P_updated)
            ps_Q = np.copy(self.sUKF.Q)
            #main
            self.sUKF.iteration(i,x,v,self.pUKF.z_updated, **kwargs)
            self.pUKF.iteration(i,x,v,ps_z_updated,ps_P_updated+ps_Q,**kwargs)