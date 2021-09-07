# Author: Seongchun Yang
# Affiliation: Kyoto University


import numpy as np
from copy import copy, deepcopy
import traceback
from KF.UKF import UnscentedKalmanFilter

class stateUKF(UnscentedKalmanFilter):
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
        Q, 
        R
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.zs     =   np.empty([n,dim_z])
        self.xps    =   np.empty([n,dim_x])
        self.Ps     =   np.empty([n,dim_z,dim_z])
        self.Pzxs   =   np.empty([n,dim_z,dim_x])
        self.Ss     =   np.empty([n,dim_x,dim_x])

        # stats
        self.innovations     =  np.empty([n,dim_x])
        self.log_likelihoods =  np.empty(n)

    def reparameterization(self, x1, p0, **kwargs):
        '''
        (OPTIONAL)
        This is not necessary unless the user wants to reparametrize the initial state.
        Reparameterization is recommended for cases where initial guess of state value and variance
        is uncertain. The reparameterization allows a single cycle of the filter to occur such that
        the updated state statistics are used in subsequent iterations as the first guess.
        
        This is using the case[1] in initiation.
        
        Parameters
        ----------
        x1   :   array_like or float
            first observation
        p0   :   array_like or float
            initial guess of parameter
        '''
        try:
            ## predict
            # ---------------------------
            self.predict(
                **{**kwargs,'past_p':p0}
            )
            
            ## update
            # ---------------------------
            self.update(
                x = x1,
                **{**kwargs,'past_p':p0}
            )
            self.post_update()
        except:
            print('Initial parameter adjustment in state filter failed.')
            raise

    def iteration(self, i, x, past_p, **kwargs):
        '''
        Normal cycle of state filter.
        IMPORTANT to note that the parameter input is the (past) updated value.
        Parameters
        ----------
        i   :   int
            iteration number
        x   :   array_like or float
            observation
        p   :   array_like or float
            **past** parameter value
        '''
        # params
        try:
            ## predict
            # ---------------------------
            self.predict(
                **{**kwargs,'past_p':past_p}
            )
            
            ## update
            # ---------------------------
            self.update(
                x = x,
                **{**kwargs,'past_p':past_p}
            )
            self.post_update()
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_c_c
            self.xps[i,:]    =   self.x_c_p
            self.Ps[i,:,:]   =   self.Pzz_c_c
            self.Pzxs[i,:,:] =   self.Pzx_c_p
            self.Ss[i,:,:]   =   self.Pxx_c_p
            # stats
            self.innovations[i,:]       =   self.innovation.flatten()
            self.log_likelihoods[i]     =   self.log_likelihood
        except:
            print('{}th iteration in state filter threw an error.'.format(i))
            raise

class parameterUKF(UnscentedKalmanFilter):
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
        Q, 
        R
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.zs     =   np.empty([n,dim_z])
        self.xps    =   np.empty([n,dim_x])
        self.Ps     =   np.empty([n,dim_z,dim_z])
        self.Pzxs   =   np.empty([n,dim_z,dim_x])
        self.Ss     =   np.empty([n,dim_x,dim_x])

        # stats
        self.innovations     =  np.empty([n,dim_x])
        self.log_likelihoods =  np.empty(n)

    def reparameterization(self, x1, z0, **kwargs):
        '''
        (OPTIONAL)
        This is not necessary unless the user wants to reparametrize the initial state (for parameter filter).
        Reparameterization is recommended for cases where initial guess of state value and variance
        is uncertain. The reparameterization allows a single cycle of the filter to occur such that
        the updated state statistics are used in subsequent iterations as the first guess.
        
        This is using the case[1] in initiation.
        
        Parameters
        ----------
        x1   :   array_like or float
            first observation
        z0   :   array_like or float
            initial guess of state filter estimate
        '''
        try:
            ## predict
            # ---------------------------
            self.predict(
                **{**kwargs,'past_z':z0}
            )
            
            ## update
            # ---------------------------
            self.update(
                x = x1,
                **{**kwargs,'past_z':z0}
            )
            self.post_update()
        except:
            print('Initial parameter adjustment in parameter filter failed.')
            raise

    def iteration(self, i, x, past_z, **kwargs):
        '''
        Normal cycle of parameter filter.
        IMPORTANT to note that the state filter input is the (past) updated value.
        Parameters
        ----------
        i   :   int
            iteration number
        x   :   array_like or float
            observation
        z   :   array_like or float
            **past** state value
        '''
        try:
            ## predict
            # ---------------------------
            self.predict(
                **{**kwargs,'past_z':past_z}
            )
            
            ## update
            # ---------------------------
            self.update(
                x = x,
                **{**kwargs,'past_z':past_z}
            )
            self.post_update()
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_c_c
            self.xps[i,:]    =   self.x_c_p
            self.Ps[i,:,:]   =   self.Pzz_c_c
            self.Pzxs[i,:,:] =   self.Pzx_c_p
            self.Ss[i,:,:]   =   self.Pxx_c_p
            # stats
            self.innovations[i,:]       =   self.innovation.ravel()
            self.log_likelihoods[i]     =   self.log_likelihood
        except:
            print('{}th iteration in parameter filter threw an error.'.format(i))
            raise

class DualUKF(object):
    '''
    This nested class is here for convenience. All variables will be saved automatically per filter and will be
    accessible readily.
    If however, nested objects are not ideal for your workflow, simply take note of the structure used and create
    your own function, initializing both the state filter and the parameter filter.

    Workflow [1] - No reparameterization
    -----------------------------------
    DUKF = DualUKF(*args)
    DUKF.main(xs)
    Workflow [2] - reparameterization
    -----------------------------------
    DUKF = DualUKF(*args)
    DUKF.reparameterization(x1)
    DUKF.main(xs)
    '''
    def __init__(
        self, 
        n,
        s_dim_z,
        s_dim_x,
        s_z0,
        s_P0,
        s_Q,
        s_R,
        s_fx,
        s_hx,
        s_points_fn,
        p_dim_z,
        p_dim_x,
        p_z0,
        p_P0,
        p_Q,
        p_R,
        p_fx,
        p_hx,
        p_points_fn
        ):
        # Force clean slate (sanitycheck)
        if hasattr(self, 'sUKF'):
            delattr(self, 'sUKF')
        if hasattr(self, 'pUKF'):
            delattr(self, 'pUKF')

        # Initiate - State Filter
        self.sUKF = stateUKF(
            n           =   n,
            dim_z       =   s_dim_z,
            dim_x       =   s_dim_x,
            z0          =   s_z0,
            P0          =   s_P0,
            fx          =   s_fx,
            hx          =   s_hx,
            points_fn   =   s_points_fn,
            Q           =   s_Q,
            R           =   s_R
        )
        self.sUKF.z_c_c = s_z0
        # Initiate - Parameter Filter
        self.pUKF = parameterUKF(
            n           =   n,
            dim_z       =   p_dim_z,
            dim_x       =   p_dim_x,
            z0          =   p_z0,
            P0          =   p_P0,
            fx          =   p_fx,
            hx          =   p_hx,
            points_fn   =   p_points_fn,
            Q           =   p_Q,
            R           =   p_R
        )
        self.pU
    
    def reparameterization(self, x1, m1 = None, **kwargs):
        '''
        (OPTIONAL)
        Find the optimal initialization point for both of the filters.
        Used when initial guess for state mean and variance are uncertain from the data.
        Parameters
        ----------
        x1  :   array_like
            first observation
        z0  :   array_like
            guess of first state value
        p0  :   array_like
            guess of first parameter value
        '''
        # we save the past values of each filter
        past_z = np.copy(self.sUKF.z_c_c)
        past_p = np.copy(self.pUKF.z_c_c)

        # Find the optimal starting point
        self.sUKF.reparameterization(
            x1 = x1,
            p0 = past_p,
            **{**kwargs,'m':m1}
        )
        self.pUKF.reparameterization(
            x1 = x1,
            z0 = past_z,
            **{**kwargs,'m':m1}
        )

    def main(self, xs, ms = None, **kwargs):
        '''
        Run all iterations for DUKF
        Parameters
        ----------
        xs  :   array_like
            example shape = (# of time steps, # of observations/sensor readings)
        ms  :   array_like
            Other inputs incorporated into kwargs for each iteration
            2D shape for iteration purposes when being provided here.
            The actual input shape for iterations will be flattend.
            As fx,hx and other functions rely on minimial explicit inputs but on kwargs,
            ms is allowed here as an explicit mention such that kwargs can recognize it and users
            can freely modifiy it in fx and hx.

        '''
        for i in range(xs.shape[0]):
            # we save the past values of each filter
            past_z = np.copy(self.sUKF.z_c_c)
            past_p = np.copy(self.pUKF.z_c_c)
            # main
            if ms is not None:
                self.sUKF.iteration(i,xs[i,:],past_p,**{**kwargs,'m':ms[i,:]})
                self.pUKF.iteration(i,xs[i,:],past_z,**{**kwargs,'m':ms[i,:]})
            else:
                self.sUKF.iteration(i,xs[i,:],past_p,**kwargs)
                self.pUKF.iteration(i,xs[i,:],past_z,**kwargs)