# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# This script is largely an adaptation of the paper DOI:10.3390/s18030808 (Zheng et al., Sensors, 2018).
# This paper was chosen as the basis for this script due to its relatively straight forward implementation
# and cost-effectiveness of computation.
# 2.
# Note that residual methods such as this requires a post-hoc adjustments to R separate from the main UKF. 
# For brevity, recompute of the state mean and variance were skipped. This also does mean that the effect 
# of the current time innovation (or prediction error) will start affecting the filter one time step after 
# the fact.
# 3.
# This method belongs to a class of 'residual based adaptive method'.
# For more information on different types of adaptation possible, see DOI:10.1109/TAC.1972.1100100.
# For more information on derivation of 'residual based adaptive method', see DOI:10.1007/s001900050236.
# 4.
# Similar implementations of adaptive filters can be found in a variety of literature;
#   DOI: 10.3390/app9091726 (Battery Health)*
#   DOI: 10.1007/s001900050236 (GPS/INS)
#   *:A variation where hyperparameter is used in place describing noise as a distribution initself is also found

import numpy as np
from copy import copy, deepcopy

class AdaptiveUnscentedKalmanFilter:
    '''
    A mixin component.
    This is not meant as a standalone filter.
    In order to utilize this as a full adaptive filter, do the following;
        Example (with UKF):
        class constructor_AUKF(UnscentedKalmanFilter,AdaptiveUnscentedKalmanFilter,metaclass=MixedClassMeta):
            def __init__(self,*args,**kwargs): pass

        AUKF = constructor_AUKF(**kwargs)

    Parameters
    ----------
    kwargs  :   dict
        + n    :   number of iterations
        + delta:   adaptive rate of filter
    '''
    def __init__(self, **kwargs):
        self.n                  =   kwargs['n']
        self.delta              =   kwargs['delta']
        self.residuals          =   np.empty((self.n,self._dim_x))
        self.residual_variances =   np.empty((self.n,self._dim_x,self._dim_x))
    
    def adapt_noise(self, i, x, **kwargs):
        '''
        Post-hoc adaptive measurement noise.
        Computes residuals which are used to construct a guaranteed positive definite matrix.
        Parameters
        ----------
        i   :   int
            iteration index
        x   :   array_like
            observation
        '''
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