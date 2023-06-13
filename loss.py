# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:47:22 2023

@author: Joana Rocha
"""

import torch
import torch.nn as nn

def STERN_loss(output, target, scaling_factor_term=None):
    if scaling_factor_term is None: 
        # initial training stage
        criterion = torch.nn.BCELoss()
        loss = criterion(output, target.float())
    
    else:
        # fine tuning training stage, considering a scaling factor term
        R = nn.ReLU()
        scaling_factor_term = R(scaling_factor_term)  
        criterion = torch.nn.BCELoss(reduction='none')
        loss = (criterion(output, target.float()) + scaling_factor_term).mean()
    
    return loss