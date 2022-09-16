import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .resnet_utilities import ResidualBlock,WNConv2d



class CouplingLayer(nn.Module):
    """Implementation of Coupling layer based on realNVP paper
    """
    def __init__(self,in_channels,mid_channels,mask_is_checkerboard,reverse_mask,num_blocks=8,use_resnet=True):
        """
        Args:
            in_channels (int): number of channels in input(image or latent space) data
            mid_channels (int): number of learnable parameters(s,t) channels
            mask_is_checkerboard (bool): True if mask is checkerboard, False if channel_wise
            reverse_mask (bool): usefull when  Combining coupling layers
            num_blocks (int): Number of blocks in resnet structure(based on the paper, for CelebA, we could use 2 blocks for residual structure)
            use_resnet (bool): If use_resnet == True , then the residual model will be resnet, otherwise it will be res_unet
        """
        super(CouplingLayer, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.mask_is_checkerboard = mask_is_checkerboard
        self.reverse_mask = reverse_mask,
        self.use_resnet = use_resnet
        
        if mask_is_checkerboard==False:
            # Base on Figure3 in paper, we should divide channels into two parts for channelwise mask
            if reverse_mask:
                in_channels = in_channels - (in_channels//2)
            else:
                in_channels = in_channels
        
        if use_resnet:
            self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                                num_blocks=num_blocks, kernel_size=3, padding=1,
                                double_after_norm=(self.mask_is_checkerboard == 1))
        else:
            self.st_net = Res_UNET(input_channels=in_channels,
                                    middle_channels=mid_channels,
                                    output_channels=mid_channels*2)
        
        
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        
    def forward(self,x,sldj=None, reverse=False):
        if self.mask_is_checkerboard:
            # Checkerboard mask
            mask = checkerboard_mask(x, start_with_one = self.reverse_mask, device=x.device)
            x_b = x * mask
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - mask)
            t = t * (1 - mask)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)
        else:
            # Channelwise mask
            if self.reverse_mask:
                x_pass_from_net,x_multiplied_with_s_and_t = x.chunk(2, dim=1)
            else:
                x_multiplied_with_s_and_t,x_pass_from_net = x.chunk(2, dim=1)
            
            s_and_t = self.st_net(x_pass_from_net)
            s,t = s_and_t.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            
            
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_multiplied_with_s_and_t = x_multiplied_with_s_and_t * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_multiplied_with_s_and_t = (x_multiplied_with_s_and_t + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)
                
            if self.reverse_mask:
                x = torch.cat((x_pass_from_net,x_multiplied_with_s_and_t), dim=1)
            else:
                x = torch.cat((x_multiplied_with_s_and_t,x_pass_from_net), dim=1)
                
            return x, sldj
        
        
        
def squeeze(x,alter_order=False,reverse=False):
    
    """
        NOT in reverse mode!
        In general, this function converts images of size (s x s x c) into a tensor with size of (s/2  x  s/2  x  4c)
        
        
        so in direct form of squeeze , width and height of image, must be dividable by 2
        in indirect(reverse) form of squeeze, channels must be dividable by 4
        
        In fact: For each spatial position, a sub-volume of shape `1x1x(N^2 * C
                 reshape into a sub-volume of shape `NxNxC`, where `N = block_size
        ------------------------------------------------

    Args:
        x (Tensor): input tensor
        alter_order (bool): Whether to use alternate ordering
    """
    if alter_order:
        batch_size,num_channels,height,width = x.shape
        if reverse:
            assert num_channels%4 == 0 , "as you can see in documentation of this function, in indirect(reverse) form of squeeze, channels must be dividable by 4"
            num_channels //= 4
        else:
            assert width%2 == 0 , "as you can see in documentation of this function, in direct form of squeeze , width of image, must be dividable by 2"
            assert height%2 == 0 , "as you can see in documentation of this function, in direct form of squeeze , height of image, must be dividable by 2"
            
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * num_channels, num_channels, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(num_channels):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
            
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(num_channels)]
                                        + [c_idx * 4 + 1 for c_idx in range(num_channels)]
                                        + [c_idx * 4 + 2 for c_idx in range(num_channels)]
                                        + [c_idx * 4 + 3 for c_idx in range(num_channels)])
        
        # shuffle_channels : tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            
            #conv2d: if x.shape == 64x3x64x64 , then perm_weight.shape == 12x3x2x2 --> x will be 64x12x32x32
            x = F.conv2d(x, perm_weight, stride=2)
            
    else:
        # if altername ordering is false
        if reverse:
            batch_size,num_channels,height,width = x.shape
            assert num_channels%4 == 0 , "as you can see in documentation of this function, in indirect form of squeeze , number of channels of images, must be dividable by 4"

            # Change format of tensor from b,c,h,w into b,h,w,c
            x = x.permute(0, 2, 3, 1)
            num_channels
            
            #Change format of tensor from b,h,w,c into b,h,w,c_,2,2
            # c_ refers to c//4
            x = x.view(batch_size,height,width,num_channels//4,2,2)

            # Change format of tensor from b,h,w,c_,2,2 into b,h,2,w,2,c_
            x = x.permute(0,1,4,2,5,3)

            # Change format of tensor from b,h,2,w,2,c_ into b,hx2,wx2,c_
            x = x.contiguous().view(batch_size,height*2,width*2,num_channels//4)

            #Finnaly change format of tensor from b,hx2,wx2,c_ into b,c_,hx2,wx2
            x = x.permute(0,3,1,2)
        else:
            
            batch_size,num_channels,height,width = x.shape
            
            assert width%2 == 0 , "as you can see in documentation of this function, in direct form of squeeze , width of image, must be dividable by 2"
            assert height%2 == 0 , "as you can see in documentation of this function, in direct form of squeeze , height of image, must be dividable by 2"
            
            

            # Change format of tensor from b,c,h,w into b,h,w,c
            x = x.permute(0, 2, 3, 1)

            x = x.view(batch_size,height//2,2,width//2,2,num_channels)

            # Change format of tensor from b,h_,2,w_,2,c into b,h_,w_,c,2,2
            x = x.permute(0,1,3,5,2,4)

            # Change format of tensor from b,h_,2,w_,2,c into b,h_,w_,4xc
            x = x.contiguous().view(batch_size,height//2,width//2,4*num_channels)

            #Finnaly change format of tensor from b,h_,w_,4xc into b,4xc,h_,w_
            x = x.permute(0,3,1,2)
    return x


def checkerboard_mask(x,device=None, requires_grad=False,start_with_one=True):
    batch_size,in_channels,height,width = x.shape
    mask = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if (i+j)%2 ==0:
                mask[i,j] = 1
                
    if start_with_one == False:
        mask = 1-mask
    
    mask = torch.tensor(mask,dtype=torch.float32, device=device, requires_grad=requires_grad)
    # Reshape from (height, width) into (1, 1, height, width) ---> for broadcasting with tensors of shape (B, C, H, W)
    return mask.view(1,1,height,width)


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
    
    
    
    
class ResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
        double_after_norm (bool): Double input after input BatchNorm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, double_after_norm):
        super(ResNet, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_norm(x)
        if self.double_after_norm:
            x *= 2.
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
    

class Res_UNET(nn.Module):
    def __init__(self,input_channels,middle_channels,output_channels):
        super(Res_UNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #######################   Encoder 1 ###########################
        
        self.encoder1 = Res_UNET.encoder_decoder(input_channels,middle_channels)
        self.conv1_1_e1 = nn.Conv2d(input_channels, middle_channels, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 2 ###########################
        
        self.encoder2 = Res_UNET.encoder_decoder(middle_channels,middle_channels*2)
        self.conv1_1_e2 = nn.Conv2d(middle_channels, middle_channels*2, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 3 ###########################
        
        self.encoder3 = Res_UNET.encoder_decoder(middle_channels*2,middle_channels*4)
        self.conv1_1_e3 = nn.Conv2d(middle_channels*2, middle_channels*4, kernel_size=(1,1), stride=(1, 1))
        #######################   Encoder 4 ###########################
        
        self.encoder4 = Res_UNET.encoder_decoder(middle_channels*4,middle_channels*8)
        self.conv1_1_e4 = nn.Conv2d(middle_channels*4, middle_channels*8, kernel_size=(1,1), stride=(1, 1))
        #######################   Bottleneck ###########################
        
        self.bottleneck = Res_UNET.encoder_decoder(middle_channels*8,middle_channels*16)
        self.conv1_1_b = nn.Conv2d(middle_channels*8, middle_channels*16, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 4 ###########################
        
        self.upconv4 = nn.ConvTranspose2d(middle_channels*16,middle_channels*8, kernel_size=(2, 2), stride=(2, 2))
        self.decoder4 = Res_UNET.encoder_decoder(middle_channels*16,middle_channels*8)
        self.conv1_1_d4 = nn.Conv2d(middle_channels*16, middle_channels*8, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 3 ###########################
        
        self.upconv3 = nn.ConvTranspose2d(middle_channels*8,middle_channels*4, kernel_size=(2, 2), stride=(2, 2))
        self.decoder3 = Res_UNET.encoder_decoder(middle_channels*8,middle_channels*4)
        self.conv1_1_d3 = nn.Conv2d(middle_channels*8, middle_channels*4, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 2 ###########################
        self.upconv2 = nn.ConvTranspose2d(middle_channels*4,middle_channels*2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder2 = Res_UNET.encoder_decoder(middle_channels*4,middle_channels*2)
        self.conv1_1_d2 = nn.Conv2d(middle_channels*4, middle_channels*2, kernel_size=(1,1), stride=(1, 1))
        #######################   Decoder 1 ###########################
        self.upconv1 = nn.ConvTranspose2d(middle_channels*2,middle_channels*1, kernel_size=(2, 2), stride=(2, 2))
        self.decoder1 = Res_UNET.encoder_decoder(middle_channels*2,middle_channels*1)
        self.conv1_1_d1 = nn.Conv2d(middle_channels*2, middle_channels, kernel_size=(1,1), stride=(1, 1))
        ###############   Last Convolutional Layer ####################
        self.last_conv = nn.Conv2d(middle_channels, output_channels, kernel_size=(1,1), stride=(1, 1))
        
    def forward(self,x):
        e1 = F.relu(self.encoder1(x) + self.conv1_1_e1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)) + self.conv1_1_e2(self.pool(e1)))
        e3 = F.relu(self.encoder3(self.pool(e2)) + self.conv1_1_e3(self.pool(e2)))
        e4 = F.relu(self.encoder4(self.pool(e3)) + self.conv1_1_e4(self.pool(e3)))
        b = F.relu(self.bottleneck(self.pool(e4)) + self.conv1_1_b(self.pool(e4)))
        d4 = self.upconv4(b)
        d4 = torch.cat((e4,d4),dim=1)
        d4 = F.relu(self.decoder4(d4) + self.conv1_1_d4(d4))
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3,d3),dim=1)
        d3 = F.relu(self.decoder3(d3) + self.conv1_1_d3(d3))
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2,d2),dim=1)
        d2 = F.relu(self.decoder2(d2) + self.conv1_1_d2(d2))
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1,d1),dim=1)
        d1 = F.relu(self.decoder1(d1) + self.conv1_1_d1(d1))
        x = self.last_conv(d1)
        return torch.sigmoid(x)
    
    @staticmethod
    def encoder_decoder(input_features,init_features):
        return nn.Sequential(
            nn.Conv2d(input_features, init_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_features, init_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(init_features)
                            )
        
