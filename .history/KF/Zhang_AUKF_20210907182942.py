# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# As close of an exact implementation of doi: 10.1109/ICIECS.2009.5365064 (Zhang et al., IEEE, 2009).
# 2.
# Reason behind the name 'forgetting scale' parameter is the author's motivation behind this implemnetation,
# which is that nonlinearity cause errors (untracked values) that accumulate over time in the filter.
# 

import numpy as np
from numpy import dot
from copy import copy, deepcopy
from KF.

class AdaptiveUnscentedKalmanFilter(object):
    def __init__(
        self, 
        dim_z, 
        dim_x, 
        z0,
        P0,
        fx, 
        hx, 
        points_fn, 
        Q, 
        R, 
        b
        ):
        super().__init__(dim_z, dim_x, z0, P0, fx, hx, points_fn, Q, R)
        self.b = b # forgetting scale (0<b<1)

    def adaptive_update(self, x, i, **kwargs):
        '''
        As is evident, both Q and R adjustments are made using innovation.
        The formulation below makes it so that we can't sufficiently guarantee that this will be PD.
        Further adaption in future for this is required for stability.
        '''
        self.d = (1-self.b)/(1-self.b**(i))
        self.R = self.R - self.d * (np.outer(self.innovation, self.innovation) - self.Pxx_c_p)
        self.Q = self.Q + self.d * (
            np.outer(
                dot(self.K, self.innovation),
                dot(self.K, self.innovation)
            ) + self.P - self.P_c_p
        )
        
    