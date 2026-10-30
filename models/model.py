import numbers
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
import torch.nn.functional as F

class ScaledResidual(nn.Module):
    """Wrap a block with a learnable residual scaling gamma.
    init: initial value for gamma (float).
    """
    def __init__(self, block, init: float = 0.2):
        super().__init__()
        self.block = block
        self.gamma = nn.Parameter(torch.ones(1) * float(init))

    def forward(self, x):
        return x + self.gamma * self.block(x)



def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features) # 深度卷积
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)# 扩张率为3，即在2个像素之间空2个像素
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x) #5*5
        attn = self.conv_spatial(attn) #7*7
        attn = self.conv1(attn) #1*1
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU() # GELU 的输出介于 ReLU（仅保留正值）和线性激活之间，负值区域有平滑的非零输出，形如高斯分布的累积函数
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        # 对每个像素的 d_model 个通道值进行线性组合，作用是融合通道间信息
        # 为后续的空间注意力提供更丰富的上下文信息，
        # 通过学习权重，增强关键通道的贡献，抑制无关通道的影响
        # proj_1 提供了一个线性变换后的特征图，使 GELU 能够基于更优化的特征分布进行非线性映射
        # 现代 CNN 架构中（如 LKABlock），通常在注意力机制前使用 1×1 卷积（设计惯例）
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        # 强调了空间维度上的关键区域，但可能导致通道间的特征分布不均衡。proj_2 通过通道间融合，重新平衡特征
        # 注意力机制后通常跟随一个前馈网络（FFN），而 FFN 常以 1×1 卷积开始和结束(设计惯例)
        x = self.proj_2(x)
        # 残差连接
        x = x + shorcut
        return x

def dwconv(ch, k, d=1):
    p = ((k - 1) // 2) * d
    return nn.Conv2d(ch, ch, k, stride=1, padding=p, dilation=d, groups=ch, bias=True)

class BigKernelMixer(nn.Module):
    """
    三个分支：21×21(d=1), 11×11(d=2)≈等效21, 7×7(d=3)≈等效19
    用softmax门控自适应加权，再用1×1融合并残差。
    建议替换你模型里“最大核那一层”的DWConv块（仅底部两层）。
    """
    def __init__(self, ch):
        super().__init__()
        self.b1 = dwconv(ch, 21, d=1)
        self.b2 = dwconv(ch, 11, d=2)
        self.b3 = dwconv(ch, 7,  d=3)
        self.fuse = nn.Conv2d(ch, ch, 1, bias=True)
        mid = max(8, ch // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(ch, mid, 1), nn.GELU(),
            nn.Conv2d(mid, 3, 1)  # 输出3个权重通道
        )

    def forward(self, x):
        y1, y2, y3 = self.b1(x), self.b2(x), self.b3(x)
        w = torch.softmax(self.gate(x), dim=1)  # [B,3,H,W]
        y = w[:,0:1]*y1 + w[:,1:2]*y2 + w[:,2:3]*y3
        return self.fuse(y) + x

class SpatialAttention2(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = BigKernelMixer(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


# 增强多尺度特征表示中对中程关系的关注
class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4., # MLP（多层感知机）隐藏层通道数的扩展倍数，默认为 4（隐藏层通道数为 dim * mlp_ratio）
                 drop=0., # Dropout 的概率，默认为0
                 drop_path=0., # DropPath（随机深度）的概率，用于正则化，默认为 0
                 act_layer=nn.GELU, # 激活函数，默认为 nn.GELU(高斯误差线性单元)
                 linear=False, # 是否在 MLP 中使用线性激活（如 ReLU），默认为 False
                 norm_cfg=dict(type='SyncBN', requires_grad=True)): # 归一化层的配置，默认为同步批归一化（SyncBN），可训练
        super().__init__()
        # 归一化层（批归一化）
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim) # 实例化SpatialAttention(dim)空间注意力，1*1，5*5，7*7组合
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity() # nn.Identity，这是一个“空操作”模块，直接返回输入

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter( # torch.ones((dim))：生成一个形状为 [dim] 的张量，所有元素为 1，乘0.01是为了初始化权重
            layer_scale_init_value * torch.ones((dim)), requires_grad=True) # requires_grad=True 表示该张量的梯度需要被计算和存储
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):     # unsqueeze(-1)用于在张量的指定维度（最后一个）上增加一个维度（大小为 1）
        # 层缩放*多核组合的空间注意力下的特征图
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) #[dim,1,1]
                               * self.attn(self.norm1(x)))  # 为了让两者能维度匹去相乘  广播机制
        # 层缩放*mlp下的特征图
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))

        return x

class LKABlock2(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4., # MLP（多层感知机）隐藏层通道数的扩展倍数，默认为 4（隐藏层通道数为 dim * mlp_ratio）
                 drop=0., # Dropout 的概率，默认为0
                 drop_path=0., # DropPath（随机深度）的概率，用于正则化，默认为 0
                 act_layer=nn.GELU, # 激活函数，默认为 nn.GELU(高斯误差线性单元)
                 linear=False, # 是否在 MLP 中使用线性激活（如 ReLU），默认为 False
                 norm_cfg=dict(type='SyncBN', requires_grad=True)): # 归一化层的配置，默认为同步批归一化（SyncBN），可训练
        super().__init__()
        # 归一化层（批归一化）
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention2(dim) # 实例化SpatialAttention(dim)空间注意力，1*1，5*5，7*7组合
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity() # nn.Identity，这是一个“空操作”模块，直接返回输入

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter( # torch.ones((dim))：生成一个形状为 [dim] 的张量，所有元素为 1，乘0.01是为了初始化权重
            layer_scale_init_value * torch.ones((dim)), requires_grad=True) # requires_grad=True 表示该张量的梯度需要被计算和存储
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):     # unsqueeze(-1)用于在张量的指定维度（最后一个）上增加一个维度（大小为 1）
        # 层缩放*多核组合的空间注意力下的特征图
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) #[dim,1,1]
                               * self.attn(self.norm1(x)))  # 为了让两者能维度匹去相乘  广播机制
        # 层缩放*mlp下的特征图
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))

        return x

# 下采样：实际上是宽和高缩小到原来的二分之一，通道数变为原来的两倍
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.PixelUnshuffle(2), # 此操作将输入张量的空间维度（高度和宽度）缩小 2 倍，同时将通道数增加 4 倍
                                  nn.Conv2d(n_feat * 2 * 2, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))
                                  # 接收 PixelUnshuffle 的输出（通道数为n_feat * 4），并将通道数减少到 n_feat * 2
                                  # 3*3的卷积核，padding＝1，使得空间维度不变
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0]) #（左，右，下，上）的位置填充0
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# 前馈网络
# 通过拓展通道，捕捉一些显著特征，通过门阀机制进一步强化关键特征，抑制无关特征
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * 3)

        # 1乘1卷积，主要用于拓展通道数，增强特征表达
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 三乘三的 深度卷积 在空间维度（H, W）上提取局部特征，每个通道独立处理，填充为一，使得卷积后尺寸不变
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)


        # 压缩通道变回36
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 转为原来的六倍的通道数
        x = self.project_in(x)
        # 深度卷积（groups=hidden_features * 2）意味着每个通道有自己的 3x3 卷积核，不与其他通道共享参数
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # chunk分割，将输出张量沿通道维度分成两部分 x1 和 x2
        # # x1和x2通道数都为原来的三倍，
        # 类似门阀ReLU 确保 x1 只贡献正值特征，配合 x2 的门控作用，动态选择哪些特征
        # x2 的值（可能正负）控制 F.relu(x1) 的输出强度：
        # 当 x2 为大正值，F.relu(x1) 被放大，增强关键特征。
        # 当 x2 接近 0，F.relu(x1) 被抑制，减少无关特征的贡献。
        # 当 x2 为负值，F.relu(x1) 被反转，可能进一步抑制或改变特征方向（视任务需求）。
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


# Frequency-Domain Pixel Attention
class FDPA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.conv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        # 可学习参数
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1)) # 初始值为0
        self.beta = nn.Parameter(torch.ones(dim, 1, 1)) #初始值为1

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # FFT in safe FP32 region for numerical stability under AMP
        with torch.cuda.amp.autocast(enabled=False):
            x1f = x1.float()
            x2f = x2.float()
            out = x1f * torch.fft.fft2(x2f)
            out = torch.fft.ifft2(out, dim=(-2, -1))
            out = torch.abs(out)
        out = out.to(x.dtype)  # # 把 out 的 dtype 转成和输入 x 一样

        # alpha初始化为 0，意味着一开始频域特征的贡献很小，模型通过训练学习是否需要更多频域信息
        # beta初始化为 1，意味着默认保留所有原始特征
        return out * self.alpha + x * self.beta


class HybridDomainAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 层归一化（单个样本范围内的归一化，在所有通道的范围内，让每个通道内的的值均值为0，方差为一
        # ，但通道之间的相对差异通过权重保留）
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)

        # Frequency-Domain Pixel Attention
        self.fpa = FDPA(dim)

        # Spatial-Domain Channel Attention
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # 将任意尺寸的输入特征图自适应地压缩为1×1的特征图
        # 提取通道维度的全局信息
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # (1 ,1) 是指输出的尺寸H，W为1

    def forward(self, x):
        # frequency pixel attention
        out = self.fpa(x)

        # spatial channel attention
        # 1x1 卷积就像一个“调色板”，对每种颜色（通道）的强度进行调整，决定哪些颜色更突出，哪些需要淡化
        s_attn = self.conv(self.pool(self.norm1(out)))
        # 这里out的值是有正有负的
        out = s_attn * out

        # 这里注意到基本x乘回注意力后都要进行一个残差连接
        # 残差连接，保留原始特征
        out = x + out
        out = out + self.ffn(self.norm2(out))

        return out


# Composite Shape Convolution
class CSC_Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        # Horizontal Strip Convolution
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        # Vertical Strip Convolution
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        # Square Kernel Convolution
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        out = self.act(out)
        return self.out_conv(out)
#----------------------------------------------------------------
# Wavelet (Haar) DWT / IWT
#-----------------------------------------------------------------
class DWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [4B, C, H/2, W/2]
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=0)


class IWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [4B, C, H, W] -> [B, C, 2H, 2W]
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch = in_batch // (r ** 2)
        out_channel, out_height, out_width = in_channel, r * in_height, r * in_width

        x1 = x[0:out_batch, :, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=x.dtype, device=x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

#-------------------------------------------------------------------------------------------
# Directional HF branch (HL/LH/HH)
#-------------------------------------------------------------------------------------------
# Directional Anisotropic Convolution (DAC) Module
class HFDirectionalBlock(nn.Module):
    """
    方向性高频增强：根据子带方向选择条形/方形深度卷积 + 1x1融合
    mode: 'HL' -> horizontal (1xK); 'LH' -> vertical (Kx1); 'HH' -> square (KxK)
    """
    def __init__(self, dim: int, mode: str, k: int = 31, bias: bool = False):
        super().__init__()
        assert mode in ['HL', 'LH', 'HH']
        pad = k // 2

        if mode == 'HL':
            dw = nn.Conv2d(dim, dim, kernel_size=(1, k), padding=(0, pad), groups=dim, bias=bias)
        elif mode == 'LH':
            dw = nn.Conv2d(dim, dim, kernel_size=(k, 1), padding=(pad, 0), groups=dim, bias=bias)
        else:  # 'HH' (diagonal/complex)
            dw = nn.Conv2d(dim, dim, kernel_size=k, padding=pad, groups=dim, bias=bias)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            dw,
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差稳定


class HFEncoderDualDec(nn.Module):
    """
    高频U-Net (带跳跃 & 双解码)
    mode: 'HL' | 'LH' | 'HH'
    """
    def __init__(self, in_channels=3, dim=24, mode='HL', k=31, bias=False,
                 Downsample=None, Upsample=None, CSC_Block=None):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, 1, bias=bias)

        # Encoder
        self.enc1 = nn.Sequential(
            ScaledResidual(HybridDomainAttention(dim), init=0.1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True),
            nn.SiLU(inplace=True)
        )
        self.down1 = Downsample(dim)                                       # -> [B, dim*2, H/4, W/4]
        self.enc2 = ScaledResidual(HFDirectionalBlock(dim*2, mode=mode, k=k, bias=bias), init=0.2)

        # Bottleneck
        self.csc = CSC_Block(dim*2)
        self.hf_bottleneck = HFDirectionalBlock(dim*2, mode=mode, k=k, bias=bias)

        # Decoder
        self.dec_pre_up_l2 = HFDirectionalBlock(dim*2, mode=mode, k=k, bias=bias)
        self.up1 = Upsample(dim*2)                                                # -> [B, dim, H/2, W/2]

        self.fuse_l1 = nn.Conv2d(dim*2, dim, 1, bias=bias)  # 拼接 (d1=dim, e1=dim) = 2*dim
        self.dec_post_up_l1 = HybridDomainAttention(dim) # post-up

        # Head & residual scale
        self.head = nn.Conv2d(dim, in_channels, 1, bias=bias)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        # x: [B, 3, H/2, W/2]
        x0 = self.embed(x)

        e1 = self.enc1(x0)                # [B, dim, H/2, W/2]
        e2 = self.enc2(self.down1(e1))    # [B, dim*2, H/4, W/4]

        b  = self.csc(e2)
        b  = self.hf_bottleneck(b)

        d2 = self.dec_pre_up_l2(b)        # pre-up
        d1 = self.up1(d2)                 # [B, dim, H/2, W/2]

        d1 = torch.cat([d1, e1], dim=1)   # 拼接 skip (dim+dim=2*dim)
        d1 = self.fuse_l1(d1)
        d1 = self.dec_post_up_l1(d1)

        out = self.head(d1)               # Δ
        return x + self.alpha * out       # 子带残差


class UIR_LLBranch(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, dim=36, bias=False):
        super(UIR_LLBranch, self).__init__()

        self.input_embed = nn.Conv2d(in_channels, dim, kernel_size=1)
        # Hybrid Domain Attention(HDA混合域注意力)
        self.encoder_level1 = HybridDomainAttention(dim)

        self.down1_2 = Downsample(dim)
        # Large Kernel Attention
        self.encoder_level2 = LKABlock(int(dim * 2 ** 1))

        self.down2_3 = Downsample(int(dim * 2 ** 1)) # dim的两倍
        self.encoder_level3 = LKABlock2(int(dim * 2 ** 2)) # dim的四倍

        # Composite Shape Convolution(复合形状卷积)（瓶颈）
        self.bottleneck = CSC_Block(int(dim * 2 ** 2))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = LKABlock2(int(dim * 2 ** 2))
        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = LKABlock(int(dim * 2 ** 1))
        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = HybridDomainAttention(int(dim))

        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=1)


    def forward(self, x):
        # 放进1*1卷积中改变通道（3--》36）
        inp = self.input_embed(x)
        out_enc_level1 = self.encoder_level1(inp) #HDA

        # 下采样：宽高变成原来的二分之一，通道数变成原来的两倍（36--》72）
        # 加LKABlock模块
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #print("inp_enc_level2.shape:",inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        #print("out_enc_level2.shape:",out_enc_level2.shape)
        # 下采样，加LKABlock模块
        inp_enc_level3 = self.down2_3(out_enc_level2)
        #print("inp_enc_level3.shape:", inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        #print("out_enc_level3.shape:", out_enc_level3.shape)

        latent = self.bottleneck(out_enc_level3)

        inp_dec_level3 = cat(latent, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) #将cat后dim*2**3 -->dim*2**2
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.final_conv(out_dec_level1) + x # LL 残差

# === BEGIN PATCH: ColorToneAdapter v2 (local contrast + global color) ===
class ColorToneAdapterV2(nn.Module):
    """
    末端颜色/对比微调头（恒等初始化）：
      - 局部对比：对亮度Y做极轻的局部对比增强（depthwise 3x3 + dilated=2，零初始化，弱残差）
      - 全局颜色：基于GAP预测 3x3 CCM + per-channel gain/bias + gamma 的微小偏移（恒等中心）
    插入位置：最终RGB输出前 out = self.cta(out)
    """
    def __init__(self, strength_local=0.2, hidden=32):
        super().__init__()
        # 局部对比（在亮度Y上生成mask）
        self.dw3  = nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=True)
        self.dw3d = nn.Conv2d(1, 1, 3, padding=2, dilation=2, groups=1, bias=True)
        nn.init.zeros_(self.dw3.weight);  nn.init.zeros_(self.dw3.bias)
        nn.init.zeros_(self.dw3d.weight); nn.init.zeros_(self.dw3d.bias)
        self.strength_local = strength_local  # 0.2 默认很小的残差系数

        # 全局颜色映射（以恒等为中心的小偏移）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(3, hidden, 1, bias=True), nn.GELU(),
            nn.Conv2d(hidden, 3 + 9 + 3 + 3, 1, bias=True)  # dg(3), dCCM(9), db(3), dgamma(3)
        )
        # 使输出初值尽量接近 0（即恒等）
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

        # 固定的RGB->Y系数（BT.601）
        self.register_buffer("rgb2y", torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1))

    def _apply_ccm(self, x, ccm):  # x: B,3,H,W; ccm: B,3,3
        # 转成 (B,H,W,3)
        y = x.permute(0, 2, 3, 1)
        # 批量矩阵乘法 (B,H,W,3) × (B,3,3) -> (B,H,W,3)
        y = torch.einsum('bhwc,bcd->bhwd', y, ccm)
        return y.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # 局部对比（亮度门控，弱残差）
        Y = (x * self.rgb2y).sum(dim=1, keepdim=True)          # B,1,H,W 获得亮度Y
        m = torch.sigmoid(self.dw3(Y) + self.dw3d(Y))          # B,1,H,W，初始≈0.5
        x = x + self.strength_local * (x * (m - 0.5) * 2.0)    # 非线性局部对比，初始≈恒等

        # 全局颜色（以恒等为中心的小偏移）
        stats = self.gap(x)                                    # B,3,1,1
        t = self.mlp(stats)                                    # B,18,1,1
        dg, dccm, dbias, dgamma = torch.split(t, [3, 9, 3, 3], dim=1)

        gain  = 1.0 + 0.10 * torch.tanh(dg)                    # B,3,1,1
        bias  = 0.05 * torch.tanh(dbias)                       # B,3,1,1
        gamma = 1.0 + 0.10 * torch.tanh(dgamma)                # B,3,1,1
        ccm   = torch.eye(3, device=x.device, dtype=x.dtype).view(1,3,3) \
                + 0.05 * torch.tanh(dccm.view(-1,3,3))         # B,3,3

        y = self._apply_ccm(x, ccm)
        y = y * gain + bias
        # y = torch.clamp(y, 0.0, 1.0)   删除看看只保留末尾clamp()                        # 保证幂运算数值安全
        # y = torch.pow(y + 1e-6, gamma)
        y = torch.pow(y.clamp_min(1e-6), gamma)  # 保证底数非负且>0

        return torch.clamp(y, 0.0, 1.0)
# === END PATCH: ColorToneAdapter v2 ===

class ECA(nn.Module):
    """Efficient Channel Attention，极轻量"""
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.gap(x)              # [B, C, 1, 1]
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, C]
        y = self.conv(y).squeeze(1)    # 通道维度 (C) 上做 1D 卷积，捕捉 相邻通道之间的依赖关系  [B, C]
        y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y

class DWResBlock(nn.Module):
    """1x1扩展 → 深度卷积（多尺度膨胀）→ GELU → 1x1回投 + 残差，可选ECA。"""
    def __init__(self, ch: int = 3, expand: int = 2, dilation: int = 1, use_eca: bool = True):
        super().__init__()
        mid = ch * expand
        self.pw1 = nn.Conv2d(ch, mid, kernel_size=1, bias=True)
        self.dw  = nn.Conv2d(mid, mid, kernel_size=3, padding=dilation, dilation=dilation,
                             groups=mid, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(mid, ch, kernel_size=1, bias=True)
        self.ca  = ECA(k_size=3) if use_eca else nn.Identity()

    def forward(self, x):
        y = self.pw1(x)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.ca(y)
        return x + y

class RefinementHead(nn.Module):
    """
    IWT 后的小头升级：多块DW残差+多尺度膨胀；默认带全局残差。
    """
    def __init__(self, ch: int = 3, blocks: int = 3, expand: int = 2,
                 dilations=(1, 2, 3), use_eca: bool = True, global_skip: bool = True):
        super().__init__()
        if isinstance(dilations, int):
            dilations = [dilations] * blocks
        elif len(dilations) < blocks:
            # 不够就循环补齐
            times = (blocks + len(dilations) - 1) // len(dilations)
            dilations = (dilations * times)[:blocks]

        self.blocks = nn.Sequential(*[
            DWResBlock(ch=ch, expand=expand, dilation=d, use_eca=use_eca) for d in dilations[:blocks]
        ])
        # 末端再抛光一下（轻量 3x3），不做归一化/激活，避免破坏动态范围
        self.final = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.global_skip = global_skip

    def forward(self, x):
        y = self.blocks(x)
        y = self.final(y)
        return x + y if self.global_skip else y

# -------------------------------
# 顶层：Wavelet + PolyKernel
# 输入/输出: [B,3,H,W] in [0,1]
# -------------------------------
class UIR_PolyKernel_Wavelet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=36, bias=False, k_hf=21):
        super().__init__()
        # 小波
        self.dwt = DWT()
        self.iwt = IWT()

        # 低频 LL 主干：复用当前网络的结构（在 H/2×W/2 上运行）
        self.ll_branch = UIR_LLBranch(in_channels, out_channels, dim, bias)

        # 高频三支
        hf_dim = max(dim // 2, 16)  # 高频通道小一点更稳更省算
        self.hf_hl = HFEncoderDualDec(out_channels, dim=hf_dim, mode='HL', k=k_hf, bias=bias,
                                      Downsample=Downsample, Upsample=Upsample, CSC_Block=CSC_Block)
        self.hf_lh = HFEncoderDualDec(out_channels, dim=hf_dim, mode='LH', k=k_hf, bias=bias,
                                      Downsample=Downsample, Upsample=Upsample, CSC_Block=CSC_Block)
        self.hf_hh = HFEncoderDualDec(out_channels, dim=hf_dim, mode='HH', k=k_hf, bias=bias,
                                      Downsample=Downsample, Upsample=Upsample, CSC_Block=CSC_Block)

        # IWT 后的轻量 refine
        self.refine = RefinementHead(ch=3, blocks=3, expand=2,
                                     dilations=(1, 2, 3), use_eca=True, global_skip=True)

        self.cta = ColorToneAdapterV2()

    @staticmethod
    def _pad_to_even(x: torch.Tensor):
        # DWT 需要 H、W 为偶数，这里自动补齐并记录以便还原
        _, _, h, w = x.shape
        pad_h = (h % 2 != 0)
        pad_w = (w % 2 != 0)
        if pad_h or pad_w:
            x = F.pad(x, (0, 1 if pad_w else 0, 0, 1 if pad_h else 0), mode='reflect')
        return x, pad_h, pad_w

    @staticmethod
    def _unpad(x: torch.Tensor, pad_h: bool, pad_w: bool):
        if pad_h:
            x = x[:, :, :-1, :]
        if pad_w:
            x = x[:, :, :, :-1]
        return x

    def forward(self, x):
        # 保障 DWT 偶数尺寸
        x_in, ph, pw = self._pad_to_even(x)

        # DWT -> 四个子带
        d = self.dwt(x_in)  # [4B, 3, H/2, W/2]
        B4, C, H2, W2 = d.shape
        B = B4 // 4
        LL, HL, LH, HH = torch.chunk(d, 4, dim=0)  # 每个 [B, 3, H/2, W/2]

        # 低频：完整主干
        LL_out = self.ll_branch(LL)  # [B, 3, H/2, W/2]

        # 高频：方向性增强
        HL_out = self.hf_hl(HL)
        LH_out = self.hf_lh(LH)
        HH_out = self.hf_hh(HH)

        # IWT 合成
        y = self.iwt(torch.cat([LL_out, HL_out, LH_out, HH_out], dim=0))  # [B, 3, H, W]（含 padding）

        # 轻量 refine + 残差到输入
        y = self.refine(y + x_in)  # 只在这里做一次 +x
        y = self._unpad(y, ph, pw)
        # out = torch.clamp(y, 0.0, 1.0)  # 删去看看

        out = self.cta(y)
        return out

if __name__ == '__main__':
    from thop import profile, clever_format

    t = torch.randn(1, 3, 256, 256).cuda()
    model = UIR_PolyKernel_Wavelet().cuda()
    macs, params = profile(model, inputs=(t,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
