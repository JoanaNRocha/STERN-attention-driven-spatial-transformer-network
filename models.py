# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:13:55 2023

@author: Joana Rocha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#%%

class STN_attention_network(nn.Module):
    '''
    Allows non-isotropic scaling and translations to modify original input images for attention purposes.
    
    As published in:
    Rocha, Joana, et al. "Attention-driven Spatial Transformer Network for Abnormality Detection in Chest X-Ray Images." 
    2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS). IEEE, 2022.
    https://doi.org/10.1109/CBMS55023.2022.00051    
    '''

    def __init__(self):
        super(STN_attention_network, self).__init__()
        
        #VGG16 binary classifier
        self.clmodel = models.vgg16(pretrained=False)
        self.clmodel.classifier[6] = nn.Linear(4096, 1)
    
        '''
        Spatial transformer localization network:
        Takes the input feature map and outputs the parameters (theta) for 
        the grid transformation.
        '''
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        '''
        Regressor for the 2*3 affine matrix:
        A final regression layer to produce the transformation parameters theta.
        '''
        self.fc_loc = nn.Sequential(
            nn.Linear(18432, 32), #nn.Linear(xs.size()[1] * xs.size()[2] * xs.size()[3], 32)
            nn.ReLU(True),
            nn.Linear(32, 4) #Four final theta parameters.
        )

        '''
        Initializeation of weights/bias:
        Initialize the last linear layer weights and biases as described in the paper.
        '''
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0.1, 0, 0], dtype=torch.float)) 
        #Bias dimensions match the number of theta parameters to predict, and the values 
        #can be initialized as identity matrix or other established value.
        

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3]) 
        theta = self.fc_loc(xs)
               
        scale_x = theta[:, 0].unsqueeze(1)
        scale_y = theta[:, 1].unsqueeze(1)
        scale_mat = torch.cat((scale_x, scale_y), 1)
        translation = theta[:, 2:].unsqueeze(2)
        
        theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)
        
        grid = F.affine_grid(theta, x.size()) #Takes a B*(2*3) theta matrix and generates an affine transformation grid.
        x = F.grid_sample(x, grid) #Transforms the image using pixel locations from the grid.

        return x, theta
    
    def forward(self, x):
        x,theta=self.stn(x) #Transformed input images and corresponding theta matrices.
        h=self.clmodel(x) #Predicted classification output
        return torch.sigmoid(h) 
    
    
#%%

class STERN_attention_network(nn.Module):
    '''
    Allows non-isotropic scaling and translations to modify original input images, with a scaling factor constraint, for attention purposes.
    
    As published in:
    ((WIP))   
    '''
    
    def __init__(self, scaling_factor):
        super(STERN_attention_network, self).__init__()
        
        self.scaling_factor = scaling_factor
        
        #VGG16 binary classifier
        self.clmodel = models.vgg16(pretrained=False)
        self.clmodel.classifier[6] = nn.Linear(4096, 1)
    
        '''
        Spatial transformer localization network:
        Takes the input feature map and outputs the parameters (theta) for 
        the grid transformation.
        '''
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        '''
        Regressor for the 2*3 affine matrix:
        A final regression layer to produce the transformation parameters theta.
        '''
        self.fc_loc = nn.Sequential(
            nn.Linear(18432, 32), #nn.Linear(xs.size()[1] * xs.size()[2] * xs.size()[3], 32)
            nn.ReLU(True),
            nn.Linear(32, 4) #Four final theta parameters.
        )

        '''
        Initializeation of weights/bias:
        Initialize the last linear layer weights and biases as described in the paper.
        '''
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0.1, 0, 0], dtype=torch.float)) 
        #Bias dimensions match the number of theta parameters to predict, and the values 
        #can be initialized as identity matrix or other established value.

    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
               
        scale_x = theta[:, 0].unsqueeze(1)
        scale_y = theta[:, 1].unsqueeze(1)        
        scale_mat = torch.cat((scale_x, scale_y), 1)   
        translation = theta[:, 2:].unsqueeze(2)
        
        theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)
        
        sx_sy=torch.div(scale_x,scale_y) #sx/sy
        
        grid = F.affine_grid(theta, x.size()) #Takes a B*(2*3) theta matrix and generates an affine transformation grid.
        x = F.grid_sample(x, grid) #Transforms the image using pixel locations from the grid.

        return x, theta, sx_sy, scale_x, scale_y
    
    def forward(self, x):
        # transform the input
        x,theta,sx_sy, scale_x, scale_y=self.stn(x)
        h=self.clmodel(x)
        return torch.sigmoid(h)