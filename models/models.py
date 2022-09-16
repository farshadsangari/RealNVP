
import torch.nn as nn
import torch
import torch.nn.functional as F
from .realnvp_utilities import CouplingLayer,squeeze


class realNVP(nn.Module):
    """
    RealNVP Model implementation
    Args:
        num_scales (int): Number of scales in Multiscale structure(Multiscale structure is available in paper)
        current_scale_idx (int): index of current scale 
        input_channels (int): Number of channels of input data
        middle_channels (int): Number of s + t channels
        data_constraint (float): data constraint which applyies in preprocess step
        num_blocks (int): Number of blocks in resnet structure(based on the paper, for CelebA, we could use 2 blocks for residual structure)
        use_resnet (bool): If use_resnet == True , then the residual model will be resnet, otherwise it will be res_unet

    """ 
    def __init__(self,num_scales,current_scale_idx,input_channels,middle_channels,data_constraint,num_blocks,use_resnet):
        super(realNVP, self).__init__()
        self.num_scales = num_scales
        self.input_channels = input_channels
        self.middle_channels = middle_channels
        self.current_scale_idx = current_scale_idx
        self.data_constraint = data_constraint
        self.num_blocks = num_blocks
        self.use_resnet = use_resnet
        
        ##############       Define Coupling      #######################
        self.is_last_scale = current_scale_idx == num_scales - 1
        
        self.first_stage_coupling_in_scales_except_last_scale_idxes = nn.ModuleList([
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard=True, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet),
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard = True, reverse_mask=True,num_blocks = num_blocks,use_resnet=use_resnet),
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard = True, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet)
                                                                   ])
        
        self.first_stage_coupling_in_last_scale_idx = nn.ModuleList([
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard=True, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet),
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard = True, reverse_mask=True,num_blocks = num_blocks,use_resnet=use_resnet),
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard = True, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet),
            CouplingLayer(input_channels, middle_channels, mask_is_checkerboard = True, reverse_mask=True,num_blocks = num_blocks,use_resnet=use_resnet)
                                                                                   ])
        
        if not self.is_last_scale:
            self.second_stage_coupling_in_scales_except_last_scale_idxes = nn.ModuleList([
                CouplingLayer(4 * input_channels, 2 * middle_channels, mask_is_checkerboard = False, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet),
                CouplingLayer(4 * input_channels, 2 * middle_channels, mask_is_checkerboard = False, reverse_mask=True,num_blocks = num_blocks,use_resnet=use_resnet),
                CouplingLayer(4 * input_channels, 2 * middle_channels, mask_is_checkerboard = False, reverse_mask=False,num_blocks = num_blocks,use_resnet=use_resnet)
                                          ])


        ### we DON'T have second stage coulpling layer for last scale idx
        
        
        self.next_scale = realNVP(num_scales,current_scale_idx+1,input_channels*2,middle_channels*2,data_constraint,num_blocks,use_resnet)
        
    def forward(self,x,reverse=False):
        if self.current_scale_idx == 0:
            sldj = None
            x, sldj = self.pre_process(x,sldj,self.data_constraint)
            
        x,sldj = self.multi_scale(x,sldj,reverse)
        return x,sldj
        
    def pre_process(self,x,data_constraint):
        y = (x * 255. + torch.rand_like(x)) / 256
        y = (2 * y - 1) * data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - data_constraint).log() - data_constraint.log())
        sldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, sldj
                 
    def multi_scale(self,x,sldj,reverse):
        

        if reverse:
            if self.is_last_block:
                for coupling in reversed(self.first_stage_coupling_in_last_scale_idx):
                    x, sldj = coupling(x, sldj, reverse)
            else:
                
                # Re-squeeze -> split -> next block
                x = squeeze(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze(x, reverse=False)
                for coupling in reversed(self.second_stage_coupling_in_scales_except_last_scale_idxes):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze(x, reverse=True)

                for coupling in reversed(self.first_stage_coupling_in_scales_except_last_scale_idxes):
                    x, sldj = coupling(x, sldj, reverse)
                    
        else:
            if self.is_last_scale:
                for coupling in self.first_stage_coupling_in_last_scale_idx:
                    x,sldj = coupling(x,sldj,reverse)
            else:
                for coupling in self.first_stage_coupling_in_scales_except_last_scale_idxes:
                    x,sldj = coupling(x,sldj,reverse)
                x = squeeze(x,reverse=False)
                
                for coupling in self.second_stage_coupling_in_scales_except_last_scale_idxes:
                    x,sldj = coupling(x,sldj,reverse)
                x = squeeze(x, reverse=True)
                x = squeeze(x, reverse=False, alt_order=True)
                
                
                
                # Base on equation 13,14,15 and 16 in the realNVP paper, we suppose that half of channels had been
                # normalized after each scale(Factoring out variables At each step)
                # Also see figure 4.b
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_scale(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze(x, reverse=True, alt_order=True)
                
        return x,sldj

