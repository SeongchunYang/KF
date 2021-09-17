# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================

import numpy as np
from copy import copy, deepcopy

class stateKalmanFilter:
    '''
    Mixin component.
    This is not meant as a standalone filter.
    
    Parameters
    ----------
    (If AUKF is used as its baseclass, no other attributes are required.
     If UKF is the direct ancestor, 'n' is required.)
    n   :   int
        # of iterations
    '''
    def __init__(self, **kwargs):
        if not hasattr(self, 'n'):
            setattr(self, 'n', kwargs['n'])
        self.zs     =   np.empty([self.n,self._dim_z])
        self.xps    =   np.empty([self.n,self._dim_x])
        self.Ps     =   np.empty([self.n,self._dim_z,self._dim_z])
        self.Pzxs   =   np.empty([self.n,self._dim_z,self._dim_x])
        self.Ss     =   np.empty([self.n,self._dim_x,self._dim_x])
        self.Qs     =   np.empty([self.n,self._dim_z,self._dim_z])
        self.Rs     =   np.empty([self.n,self._dim_x,self._dim_x])

        # stats
        self.innovations     =  np.empty([self.n,self._dim_x])
        self.log_likelihoods =  np.empty(self.n)

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
            self.post_update(
                **{**kwargs,'past_p':p0}
            )
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
            self.post_update(
                **{**kwargs,'past_p':past_p}
            )
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_c_c
            self.xps[i,:]    =   self.x_c_p
            self.Ps[i,:,:]   =   self.Pzz_c_c
            self.Pzxs[i,:,:] =   self.Pzx_c_p
            self.Ss[i,:,:]   =   self.Pxx_c_p
            self.Qs[i,:,:]   =   self.Q
            self.Rs[i,:,:]   =   self.R
            # stats
            self.innovations[i,:]       =   self.innovation.flatten()
            self.log_likelihoods[i]     =   self.log_likelihood

            ## adapt noise covariance
            # ---------------------------
            if hasattr(self, 'adapt_noise') and callable(self.adapt_noise):
                self.adapt_noise(i, x, **{**kwargs,'past_p':past_p})
        except:
            print('{}th iteration in state filter threw an error.'.format(i))
            raise


class parameterKalmanFilter:
    def __init__(self, **kwargs):
        '''
        Mixin component.
        This is not meant as a standalone filter.
        
        Parameters
        ----------
        (If AUKF is used as its baseclass, no other attributes are required.
         If UKF is the direct ancestor, 'n' is required.)
        n   :   int
            # of iterations
        '''
        if not hasattr(self, 'n'):
            setattr(self, 'n', kwargs['n'])
        self.zs     =   np.empty([self.n,self._dim_z])
        self.xps    =   np.empty([self.n,self._dim_x])
        self.Ps     =   np.empty([self.n,self._dim_z,self._dim_z])
        self.Pzxs   =   np.empty([self.n,self._dim_z,self._dim_x])
        self.Ss     =   np.empty([self.n,self._dim_x,self._dim_x])
        self.Qs     =   np.empty([self.n,self._dim_z,self._dim_z])
        self.Rs     =   np.empty([self.n,self._dim_x,self._dim_x])

        # stats
        self.innovations     =  np.empty([self.n,self._dim_x])
        self.log_likelihoods =  np.empty(self.n)
    
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
            self.post_update(
                **{**kwargs,'past_z':z0}
            )
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
            self.post_update(
                **{**kwargs,'past_z':past_z}
            )
            
            ## save
            # ---------------------------
            # states
            self.zs[i,:]     =   self.z_c_c
            self.xps[i,:]    =   self.x_c_p
            self.Ps[i,:,:]   =   self.Pzz_c_c
            self.Pzxs[i,:,:] =   self.Pzx_c_p
            self.Ss[i,:,:]   =   self.Pxx_c_p
            self.Qs[i,:,:]   =   self.Q
            self.Rs[i,:,:]   =   self.R
            # stats
            self.innovations[i,:]       =   self.innovation.flatten()
            self.log_likelihoods[i]     =   self.log_likelihood

            ## adapt noise
            # ---------------------------
            if hasattr(self, 'adapt_noise') and callable(self.adapt_noise):
                self.adapt_noise(i, x, **{**kwargs,'past_z':past_z})
        except:
            print('{}th iteration in parameter filter threw an error.'.format(i))
            raise

class DualKalmanFilter:
    '''
    More of an assortment of initiated classes than an inherited object.
    In fact, initial parameters require one state and one parameter filter
    pre-initialized.
    The code is meant to be filter agnostic. As such, one can use the this
    for a variety of filters pre-initialized for use in this class, as long
    as the base classes use the same naming scheme.

    Parameters
    ----------
    sKF     :   object
        state filter
    pKF     :   object
        parameter filter
    '''
    def __init__(self, sKF, pKF):
        self.sKF    =   sKF
        self.pKF    =   pKF
    
    def initiate(self):
        # Initiate - State Filter
        self.sKF.z_c_c = self.sKF.z
        self.sKF.Pzz_c_c = self.sKF.P

        # Initiate - Parameter Filter
        self.pKF.z_c_c = self.pKF.z
        self.pKF.Pzz_c_c = self.pKF.P

    def reparameterization(self, x1, m1 = None, **kwargs):
        '''
        (OPTIONAL)
        Find the optimal initialization point for both of the filters.
        Used when initial guess for state mean and variance are uncertain from the data.
        Parameters
        ----------
        x1  :   array_like
            first observation
        m1  :   array_like
            input
        '''
        # we save the past values of each filter
        past_z = np.copy(self.sKF.z_c_c)
        past_p = np.copy(self.pKF.z_c_c)

        # Find the optimal starting point
        self.sKF.reparameterization(
            x1 = x1,
            p0 = past_p,
            **{**kwargs,'m':m1}
        )
        self.pKF.reparameterization(
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
            shape = (# of time steps, # of observations/sensor readings)
        ms  :   array_like
            shape = (# of time steps, # of input classes)
        '''
        for i in range(xs.shape[0]):
            # we save the past values of each filter
            past_z = np.copy(self.sKF.z_c_c)
            past_p = np.copy(self.pKF.z_c_c)
            # main
            if ms is not None:
                self.sKF.iteration(i,xs[i,:],past_p,**{**kwargs,'m':ms[i,:]})
                self.pKF.iteration(i,xs[i,:],past_z,**{**kwargs,'m':ms[i,:]})
            else:
                self.sKF.iteration(i,xs[i,:],past_p)
                self.pKF.iteration(i,xs[i,:],past_z)