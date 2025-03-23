import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.ops import DeformConv2d

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .polar_utils import get_sample_params_from_subdiv, get_sample_locations
# import torch.utils.checkpoint as checkpoint
import numpy as np
pi = 3.141592653589793



class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def R(window_size, num_heads, radius, D, a_r, b_r, r_max):
    # import pdb;pdb.set_trace()
    a_r = a_r[radius.view(-1)].reshape(window_size[0]*window_size[1], window_size[0]*window_size[1], num_heads)
    b_r = b_r[radius.view(-1)].reshape(window_size[0]*window_size[1], window_size[0]*window_size[1], num_heads)
    radius = radius[None, :, None, :].repeat(num_heads, 1, D.shape[0], 1) # num_heads, wh, num_win*B, ww
    radius = D*radius

    radius = radius.transpose(0,1).transpose(1,2).transpose(2,3).transpose(0,1)

    # A_r = torch.zeros(window_size[0]*window_size[1], window_size[0]*window_size[1], num_heads).cuda()
    A_r = a_r*torch.cos(radius*2*pi/r_max) + b_r*torch.sin(radius*2*pi/r_max)
    
    return A_r

def theta(window_size, num_heads, radius, theta_max, a_r, b_r, H): # change theta_max to D ====self.window_size, self.num_heads, self.radius, theta_max, self.a_r, self.b_r, self.input_resolution[0])
    a_r = a_r[radius]
    b_r = b_r[radius]
    # print("radius",radius.shape,theta_max.shape,H)#torch.Size([64, 64]) torch.Size([2]) 128
    time_t=int((window_size[0]*window_size[1])/theta_max.shape[0])
    # print("time_t",time_t,window_size[0],window_size[1])
    theta_max=theta_max.repeat_interleave(time_t, 0).flatten()
    # print("theta_max.shape",theta_max.shape)
    radius = radius*theta_max/H
    radius = radius[:, :, None].repeat(1, 1, num_heads)
    A_r = a_r*torch.cos(radius) + b_r*torch.sin(radius)
    
    return A_r

def phi(window_size, num_heads, azimuth, a_p, b_p, W):
    a_p = a_p[azimuth]
    b_p = b_p[azimuth]
    azimuth = azimuth*2*np.pi/W
    azimuth = azimuth[:, :, None].repeat(1, 1, num_heads)

    A_phi = a_p*torch.cos(azimuth) + b_p*torch.sin(azimuth)
    # import pdb;pdb.set_trace()
    return A_phi 

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # print(x.shape)
    B, H, W, C = x.shape
    # print("partition inp",x.shape)#([1, 128, 8, 192])
    # print("window_size",window_size)#(1, 16)
    # print("D_s inp",D_s.shape)#([1, 128, 8, 192])
    # import pdb;pdb.set_trace()
    if type(window_size) is tuple:
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        # D_s = D_s.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        # windows_d = D_s.permute(0, 1, 3, 2, 4).contiguous().view(-1, window_size[0], window_size[1])
        return windows
    else:
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        # D_s = D_s.view(B, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        # windows_d = D_s.permute(0, 1, 3, 2, 4).contiguous().view(-1, window_size, window_size)
        return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    
    if type(window_size) is tuple:
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        # D_s = D_windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1])
    else:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        # D_s = D_windows.view(B, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # D_s = D_s.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)
    # import pdb;pdb.set_trace()
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, patch_size,input_resolution, dim, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        # import pdb;pdb.set_trace()
        super().__init__()
        # print("window_size", window_size)
        # import pdb;pdb.set_trace()
        self.dim = dim
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        #print("attention patch size",patch_size)
        self.window_size = window_size  # Wh, Ww
        #print("attention window_size",window_size)
        self.num_heads = 1
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        H, W = input_resolution

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # self.a_p = nn.Parameter(
        #     torch.zeros(9, num_heads))
        # self.b_p = nn.Parameter(
        #     torch.zeros(8, num_heads))
        # self.a_r = nn.Parameter(
        #     torch.zeros(9, num_heads))
        # self.b_r = nn.Parameter(
        #     torch.zeros(8, num_heads))
        

        if input_resolution == window_size:
            self.a_p = nn.Parameter(
                torch.zeros(window_size[1], self.num_heads))
            self.b_p = nn.Parameter(
                torch.zeros(window_size[1], self.num_heads))
        else:
            self.a_p = nn.Parameter(
                torch.zeros((2 * window_size[1] - 1), self.num_heads))
            self.b_p = nn.Parameter(
                torch.zeros((2 * window_size[1] - 1), self.num_heads))
        self.a_r = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1), self.num_heads))
        self.b_r = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1), self.num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        #print("coords1",coords.shape)#torch.Size([2, 8, 8])
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        #print("coords_flatten",coords_flatten.shape)#torch.Size([2, 64])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww#torch.Size([2, 64, 64])
        #print("relative_coords1",relative_coords.shape)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        #print("relative_coords2",relative_coords.shape)#torch.Size([64, 64, 2])
        radius = (relative_coords[:, :, 0]).cuda()#[64, 64, 1]
        azimuth = (relative_coords[:, :, 1]).cuda()
        r_max = self.patch_size[0]*H
        # print("patch_size", patch_size[0], "azimuth", 2*np.pi/W, "r_max", r_max)
        self.r_max = r_max
        self.radius = radius
        self.azimuth = azimuth

        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=4)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.a_p, std=.02)
        trunc_normal_(self.a_r, std=.02)
        trunc_normal_(self.b_p, std=.02)
        trunc_normal_(self.b_r, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, theta_max, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or Nonem

        """

        B_, N, C = x.shape
        #print("wa x.shape",x.shape)
        # import pdb;pdb.set_trace()
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        #print("attn",attn.shape)#torch.Size([32, 1, 64, 64])
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #print("self.input_resolution",self.input_resolution)
        A_phi = phi(self.window_size, self.num_heads, self.azimuth, self.a_p, self.b_p, self.input_resolution[1])
        #print("A_phi",A_phi.shape)#torch.Size([64, 64, 1])
        A_theta = theta(self.window_size, self.num_heads, self.radius, theta_max, self.a_r, self.b_r, self.input_resolution[0]) # change input_resolution[0] to r_max
        # A_r = R(self.window_size, self.num_heads, self.radius, D, self.a_r, self.b_r, self.r_max)
        attn = attn + A_phi.transpose(1, 2).transpose(0, 1).unsqueeze(0) + A_theta.transpose(1, 2).transpose(0, 1).unsqueeze(0) 

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops




def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self,patch_size, input_resolution,dim,ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False,drop_path=0.):
        super(TransformerBlock, self).__init__()
        self.window_size=(1,input_resolution[1])
        self.input_resolution=input_resolution
        self.att = att
        if self.att:
            self.norm1 = nn.LayerNorm(dim)
            # self.attn = FSAS(dim, bias)
            # patch_size
            self.patch_size=patch_size
            self.attn = WindowAttention(self.patch_size,self.input_resolution, dim, window_size=to_2tuple(self.window_size),
            qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.attn_mask=None
        self.drop_path=DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inp):
        x=inp[0]#BCHW
        D_s=inp[1]
        theta_max=inp[2]
        # H,W=self.input_resolution
        # print("B,H,W,C",B,H,W,C)#1 128 128 48 128
        # assert N == H * W, "input feature has wrong size"
        if self.att:
            #print("attetnion cal!!")
            # x = x + self.attn(self.norm1(x))
            x=x.permute(0,2,3,1)#BHWC
            B,H,W,C=x.shape
            #print("B,H,W,C",B,H,W,C)#1 128 128 48 128
            x=x.view(B,H*W,C)
            # H, W = self.input_resolution
            
            # print(H, W, self.input_resolution, self.window_size)
            # breakpoint()
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)
            #print("x_attn",x.shape)#([1, 128, 8, 192])

            # cyclic shift
            # if self.shift_size > (0, 0):
            #     shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            # else:
            shifted_x = x

            # partition windows
            x_windows= window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            #print("x_windows",x_windows.shape)#([256, 2, 2, 192])
            if type(self.window_size) is tuple:
                x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
                # D_windows = D_windows.view(-1, self.window_size[0] * self.window_size[1])  # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                attn_windows = self.attn(x_windows, theta_max, mask=self.attn_mask)  # nW*B, window_size*window_size, C

                # merge windows
                attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
            else:
                x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
                # D_windows = D_windows.view(-1, self.window_size * self.window_size)
                # W-MSA/SW-MSA
                attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

                # merge windows
                attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

            shifted_x = window_reverse(attn_windows,self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            # if self.shift_size > (0, 0):
            #     x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            # else:
            x = shifted_x
            x = x.view(B, H * W, C)
            
            x = shortcut + self.drop_path(x)
            x=x.view(B,H,W,C).permute(0,3,1,2).contiguous()#BCHW
            
            #print("attention_down::",x.shape)#([1, 32, 32, 192])
        
        #huanyuanweidu 
        # print("xxx",len(x),len(x[0]),len(x[0][0]),len(x[0][0][0]))
        # print("xxx",x)
        #print("xxx1",x.shape)# torch.Size([1, 128, 128, 48])BHWC
        # x = x.permute(0,3,1,2)#BCHW
        # print("xxx2",x.shape)#xxx2 torch.Size([1, 16, 48, 8])
        x = x + self.ffn(self.norm2(x))
        # print("xxx2",x.shape)#torch.Size([1, 48, 8, 16])
        # x = x.permute(0,2,3,1)#B H W  C
        #print("FFNdown",x.shape)
        out=[x,D_s,theta_max]
        return out


class Fuse(nn.Module):
    def __init__(self, n_feat,l,patches_resolution,patch_size):
        super(Fuse, self).__init__()
        
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(patch_size=patch_size,input_resolution=(patches_resolution[0] // (1 ** l),
                                                 patches_resolution[1] // (4 ** l)),dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        #print("dnc[0]",enc[0].shape)#([1, 64, 64, 96])
        #print("enc[0]",dnc[0].shape)#([1, 64, 64, 96])
        _ = self.conv(torch.cat((enc[0], dnc[0]), dim=1))
        x=[_,enc[1],enc[2]]
        x = self.att_channel(x)
        _ = self.conv2(x[0])
        e, d = torch.split(_, [self.n_feat, self.n_feat], dim=1)
        output = e + d
        x=[output,enc[1],enc[2]]
        return x

def conv_polar(input_tensor, fan_regions=1, in_c=3, embed_dim=48, bias=False):
    b, c, h, w = input_tensor.size()
    start_angle = 0
    end_angle = 2 * np.pi / fan_regions
    center_x = w // 2
    center_y = h // 2
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    dx = grid_x - center_x
    dy = grid_y - center_y
    angle = torch.atan2(dy, dx)
    fan_idx = torch.floor((angle - start_angle+np.pi) / (end_angle - start_angle) ).long()
    fan_conv_results = torch.zeros(b,embed_dim,h,w,dtype=torch.float).cuda()
    softmax = nn.Softmax(dim=1)
    fan_conv=nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias).cuda()

    for i in range(fan_regions):
        # print("ii",i)
        if(i==fan_regions-1):
            mask1=(fan_idx == i)
            mask2=(fan_idx == i+1)
            mask=(mask1+mask2).cuda()
        else:
            mask = (fan_idx == i).cuda()
        fan_input = (input_tensor*mask)
        fan_conv=nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias).cuda()
        fan_conv_result = (fan_conv(fan_input)*mask).cuda()
        fan_conv_results+=fan_conv_result
    fan_conv_results = softmax(fan_conv_results).cuda()
    return fan_conv_results

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.dcn = DCN(embed_dim, embed_dim, kernel_size=(3,3), stride=1, padding=1).cuda()
        # self.deformable_conv2d = DeformableConv2d(in_dim=3, out_dim=embed_dim, kernel_size=3, padding=1,stride=1,offset_groups=3, with_mask=False)
        # self.dcon = DConv(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dconv = DeformConv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dcon_c = DConv(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        fan_regions=1
        kernel_size=3
        out2=conv_polar(x, fan_regions=fan_regions, in_c=3, embed_dim=2 * kernel_size * kernel_size, bias=False)
        out1=self.dcon_c(x)
        out2=self.dconv(x,out2)
        out=out1+out2
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        # x=x.permute(0,3,1,2)#BCHW
        x=self.body(x)
        #print("sample:::",x.shape)
        # x=x.permute(0,2,3,1)#BHWC
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        # x=x.permute(0,3,1,2)#BCHW
        x=self.body(x)
        #print("sample:::",x.shape)
        # x=x.permute(0,2,3,1)#BHWC
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, distortion_model = 'polynomial', radius_cuts=16, azimuth_cuts=64, radius=None, azimuth=None, in_chans=3, embed_dim=96, n_radius = 8, n_azimuth=16, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        
        #number of MLPs is number of patches
        #patch_size if needed
        patches_resolution = [radius_cuts, azimuth_cuts]  ### azimuth is always cut in even partition 
        self.azimuth_cuts = azimuth_cuts#64
        self.radius_cuts = radius_cuts#16
        self.subdiv = (self.radius_cuts, self.azimuth_cuts)
        self.img_size = img_size
        self.distoriton_model = distortion_model
        self.radius = radius
        self.azimuth = azimuth
        # self.measurement = 1.0
        self.max_azimuth = np.pi*2#圆周角度
        patch_size = [self.img_size[0]/(2*radius_cuts), self.max_azimuth/azimuth_cuts]
        # self.azimuth = 2*np.pi  comes from the cartesian script

        
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = radius_cuts*azimuth_cuts
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        
        # subdiv = 3
    #    print("n_radius,n_azimuth",n_radius,n_azimuth)
        self.n_radius = n_radius
        self.n_azimuth = n_azimuth
        self.mlp = nn.Linear(self.n_radius*self.n_azimuth*in_chans, embed_dim)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, dist):
        B, C, H, W = x.shape
        #print("xxx",x.shape)#([8, 3, 128, 128])

        dist = dist.transpose(1,0)
        #print("dist",dist.shape)#([4, 1])
        radius_buffer, azimuth_buffer = 0, 0
        params, D_s, theta_max = get_sample_params_from_subdiv(
            subdiv=self.subdiv,
            img_size=self.img_size,
            distortion_model = self.distoriton_model,
            D = dist, 
            n_radius=self.n_radius,
            n_azimuth=self.n_azimuth,
            radius_buffer=radius_buffer,
            azimuth_buffer=azimuth_buffer)
        # print(params)
        # print(D_s.size()) #torch.Size([1, 16, 64])
        # print(theta_max)#tensor([3.], device='cuda:1')
        # import pdb;pdb.set_trace()
        sample_locations = get_sample_locations(**params,img_B=B)  ## B, azimuth_cuts*radius_cuts, n_radius*n_azimut :2 1, 16384, 128
        #print("sample_loc:",len(sample_locations),len(sample_locations[0]),len(sample_locations[0][0]),len(sample_locations[0][0][0]))
        B2, n_p, n_s = sample_locations[0].shape#[1, 16384, 128])
        #print("sample_locations[0].shape",sample_locations[0].shape)#[64, 16384, 128])
        #print("sample_locations[1].shape",sample_locations[1].shape)#
        
        
        #n_p= subdiv[0]*subdiv[1]=radius_cuts* self.azimuth_cuts    n_s= n_radius*n_azimuth=8*16
        x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
        x_ = x_/(H//2)
        y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()
        y_ = y_/(W//2)
        out = torch.cat((y_, x_), dim = 3)
    #    print("out:",out.shape)#([2, 16384, 1, 2])
        out = out.cuda()
    #    print(out.shape)

        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        ############################ projection layer ################
        x_out = torch.empty(B, self.embed_dim, self.radius_cuts, self.azimuth_cuts).cuda(non_blocking=True)#out: torch.Size([1, 1024, 100, 2])
    #    print("x_out",x_out.shape)#([2, 48, 128, 128])
        #print("x",x.shape)# [2, 3, 128, 128])
        #print("out",out.shape)#([64, 16384, 128, 2])
        #维度为(N,C,Hin,Win) 的input，维度为(N,Hout,Wout,2) 的grid，则该函数output的维度为(N,C,Hout,Wout)
        # tensor_see = nn.functional.grid_sample(x, out, align_corners = True)
        tensor = nn.functional.grid_sample(x, out, align_corners = True).permute(0,2,1,3).contiguous().view(-1, self.n_radius*self.n_azimuth*self.in_chans)
        
        

        out_ = self.mlp(tensor)#n_radius*n_azimuth*in_chans--->embed_dim
    #    print("out_1",out_.shape)#tensor2 torch.Size([32768, 48])
        out_ = out_.contiguous().view(B, self.radius_cuts*self.azimuth_cuts, -1)   
    #    print("out_2",out_.shape)#out_ torch.Size([2, 16384, 48])

        out_up  = out_.reshape(B, self.azimuth_cuts, self.radius_cuts, self.embed_dim)  ### check the output dimenssion properly
    #    print("out_up1",out_up.shape)#torch.Size([2, 128, 128, 48])
        # out_up = torch.flip(out_up, [1])  # (B,  az_div/2, rad_div, embed dim)
        out_up = out_up.transpose(1, 3)
    #    print("out_up2.shape",out_up.shape)#torch.Size([2, 48, 128, 128])
        # out_down = out_down.transpose(1,3)
        x_out[:, :, :self.radius_cuts, :] = out_up
        x= x_out.permute(0,2,3,1).contiguous()#BHWC
        if self.norm is not None:
            x = self.norm(x)
        x=x.permute(0,3,1,2).contiguous()#BHWC
        return x, D_s, theta_max

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops



##########################################################################
##---------- mdt -----------------------
class mdt(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 img_size=128,
                 ):
        super(mdt, self).__init__()

        # self.patch_embed = PatchEmbed(inp_channels, dim)
        res=img_size
        distortion_model = 'polynomial'
        norm_layer=nn.LayerNorm
        radius_cuts=res
        azimuth_cuts = res
        n_radius=1
        n_azimuth=1
        cartesian = torch.cartesian_prod(
            torch.linspace(-1, 1, res),
            torch.linspace(1, -1, res)
        ).reshape(res, res, 2).transpose(2, 1).transpose(1, 0).transpose(1, 2)
        radius = cartesian.norm(dim=0)
        theta = torch.atan2(cartesian[1], cartesian[0])
        # self.patch_norm = True
        self.patch_embed0 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed = PatchEmbed(img_size=img_size,distortion_model = distortion_model, radius_cuts=radius_cuts, azimuth_cuts= azimuth_cuts,  radius = radius, azimuth = theta, in_chans=inp_channels, embed_dim=dim,n_radius=n_radius, n_azimuth=n_azimuth,
            norm_layer=norm_layer)
        
        drop_rate=0.
        self.ape = False
        num_patches = self.patch_embed.num_patches
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_path_rate=0.1
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]  # stochastic depth decay rule
        patches_resolution = self.patch_embed.patches_resolution 
        patch_size = self.patch_embed.patch_size
        #print("patches_resolution",patches_resolution)#[8, 16]
        #print("patch_size",patch_size)#[8, 16]
        patches_res_num=4
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 0),
                                                 patches_resolution[1] // (patches_res_num ** 0)),
                             dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:0 + 1])][i],patch_size=patch_size) for i in
            range(num_blocks[0])])
    
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 1),
                                                 patches_resolution[1] // (patches_res_num ** 1)),
                             dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:0 + 2])][i],patch_size=patch_size) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 2),
                                                 patches_resolution[1] // (patches_res_num ** 2)),
                dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:0 + 3])][i],patch_size=patch_size) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 2),
                                                 patches_resolution[1] // (patches_res_num ** 2)),
                            dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:0 + 3])][i],patch_size=patch_size) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 1),
                                                 patches_resolution[1] // (patches_res_num ** 1)),
                             dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:0 + 2])][i],patch_size=patch_size) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 0),
                                                 patches_resolution[1] // (patches_res_num ** 0)),
                             dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:0 + 1])][i],patch_size=patch_size) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(input_resolution=(patches_resolution[0] // (1 ** 0),
                                                 patches_resolution[1] // (patches_res_num ** 0)),
                             dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:0 + 1])][i],patch_size=patch_size) for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2,2,patches_resolution,patch_size)
        self.fuse1 = Fuse(dim,1,patches_resolution,patch_size)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, dist):
        # inp_enc_level1 = self.patch_embed(inp_img)

        inp_enc_level1, D_s, theta_max = self.patch_embed(inp_img,dist)
        inp_enc_level1=self.patch_embed0(inp_img)
        # print(inp_enc_level1.shape)
        # trans_img=self.output(inp_enc_level1)
        # print(trans_img)
        # print("trans_img.shape",trans_img.shape)

        #print("inp_enc_level1",inp_enc_level1.size())#[1, 48, 128, 128])
        if self.ape:
            inp_enc_level1 = inp_enc_level1 + self.absolute_pos_embed
        # trans_img=self.output(inp_enc_level1)

        inp_enc_level1 = self.pos_drop(inp_enc_level1)
        # trans_img=self.output(inp_enc_level1)

        # breakpoint()
        inp_enc1=[inp_enc_level1, D_s, theta_max]
        out_enc_level1 = self.encoder_level1(inp_enc1)
        #print("enc_level1",out_enc_level1[0].shape)#([1, 128, 128, 48])

        inp_enc_level2 = self.down1_2(out_enc_level1[0])
        inp_enc2=[inp_enc_level2, out_enc_level1[1], theta_max]
        out_enc_level2 = self.encoder_level2(inp_enc2)
        #print("enc_level2",out_enc_level2[0].shape)#torch.Size([1, 64, 64, 96])

        inp_enc_level3 = self.down2_3(out_enc_level2[0])
        inp_enc3=[inp_enc_level3, out_enc_level2[1],theta_max]
        out_enc_level3 = self.encoder_level3(inp_enc3)
        #print("enc_level3",out_enc_level3[0].shape)#torch.Size([1, 32, 32, 192])

        inp_dec3=[out_enc_level3[0], out_enc_level3[1],theta_max]
        #print("inp_dec3",out_enc_level3[0].shape)#([1, 32, 32, 192])
        out_dec_level3 = self.decoder_level3(inp_dec3)
        #print("dec_level3",out_dec_level3[0].shape)#([1, 32, 32, 192])

        inp_dec_level2 = self.up3_2(out_dec_level3[0])
        inp_dec_level2 = [inp_dec_level2, out_dec_level3[1],theta_max]
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)
        #print("Fuse!")
        inp_dec2=[inp_dec_level2[0], out_dec_level3[1],theta_max]
        out_dec_level2 = self.decoder_level2(inp_dec2)
        #print("dec_level2",out_dec_level2[0].shape)#([1, 96, 64, 64])

        inp_dec_level1 = self.up2_1(out_dec_level2[0])
        inp_dec_level1 = [inp_dec_level1, out_dec_level2[1],theta_max]
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        #print("Fuse!")
        inp_dec1=[inp_dec_level1[0], out_dec_level2[1],theta_max]
        out_dec_level1 = self.decoder_level1(inp_dec1)
        #print("dec_level1",out_dec_level1[0].shape)

        inp_dec1=[out_dec_level1[0], out_dec_level1[1],theta_max]
        out_dec_level1 = self.refinement(inp_dec1)
        #print("inp_img.shape",inp_img.shape)
        out_dec_level1 = self.output(out_dec_level1[0]) + inp_img

        return out_dec_level1



