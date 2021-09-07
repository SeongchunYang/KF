# Author: Seongchun Yang
# Affiliation: Kyoto University
# ======================================================================
# 1. (IMPORTANT:CITATION ALERT)
# As close of an exact implementation of doi: 10.1109/ICIECS.2009.5365064 (Zhang et al., IEEE, 2009).

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
        d
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

    def adaptive_update(self, x, i, **kwargs):
        '''
        As is evident, both Q and R adjustments are made using innovation.
        The formulation below makes it so that we can't sufficiently guarantee that this will be PD.
        Further adaption in future for this is required for stability.
        '''
        self.dk_1 = (1-self.d)/(1-self.d**(i))
        self.R = self.R - self.dk_1 * (np.outer(self.innovation, self.innovation) - self.S)
        self.Q = self.Q + self.dk_1 * (
            np.outer(
                dot(self.K, self.innovation), 
                dot(self.K, self.innovation)
            ) + self.P_updated - self.P_og
        )
        self.update(x, **kwargs)
        
    