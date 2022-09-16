

import torch.nn as nn
import numpy as np

class RealNVPLoss(nn.Module):
    
    """This function calculate and returns NLL loss for realNVP
        in this Function we consider that distribution in latent space follows Gaussian distribution with specific mean and Variance
        ------------------------------------------------
    Args:
        k (int): Number of discrete values in each input dimension
        distribution_mean (k): Mean of gussian distribution which we use in loss term
        distribution_std (k): Standard deviation of gussian distribution which we use in loss term

    Returns:
        Tensor(with shape of batchSize):  NLL loss for realNVP
    """
    def __init__(self,distribution_mean=0,distribution_std=1, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k
        self.distribution_mean = distribution_mean
        self.distribution_std = distribution_std

    def forward(self, z, sldj):
        # First of all, we compute log of the gaussian distribution 
        prior_ll = -0.5*(((z - self.distribution_mean)/self.distribution_std)**2) - np.log(self.distribution_std*np.sqrt(2*np.pi))
        
        # then we reshape the prior tensor from (batch_size , channels , height , width) into
        # (batch_size ,channels , height , width) then summing up the values in second dimention
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        
        # We add prior with log-determinant of Jacobian of initial transform(sldj) and then return minus of the mean of the loss in batch
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll