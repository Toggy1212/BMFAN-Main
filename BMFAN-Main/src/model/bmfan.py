import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


try:
    from basicsr.utils.registry import ARCH_REGISTRY
except ImportError:
    class _DummyRegistry:
        def register(self):
            def deco(x):
                return x
            return deco
    ARCH_REGISTRY = _DummyRegistry()

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    class PyTorchModelHubMixin:
        pass


class DFE(nn.Module):


    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0)
        )

        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, H * W).permute(0, 2, 1).contiguous()
        return x


class Mlp(nn.Module):


    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):


    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size[0], window_size[0],
        W // window_size[1], window_size[1],
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size[0], window_size[1], C
    )
    return windows


def window_reverse(windows, window_size, H, W):


    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(
        B,
        H // window_size[0], W // window_size[1],
        window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DynamicPosBias(nn.Module):


    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class VWindowSpatialGate(nn.Module):


    def __init__(self, Chead, window_size, use_asym=True):
        super().__init__()
        self.Chead = Chead
        self.window_size = window_size
        self.use_asym = use_asym

        if use_asym:
            self.dw1 = nn.Conv2d(Chead, Chead, kernel_size=(1, 3), padding=(0, 1),
                                 groups=Chead, bias=True)
            self.dw2 = nn.Conv2d(Chead, Chead, kernel_size=(3, 1), padding=(1, 0),
                                 groups=Chead, bias=True)
            self.pw = nn.Conv2d(Chead, Chead, kernel_size=1, bias=True)
        else:
            self.dw = nn.Conv2d(Chead, Chead, kernel_size=3, padding=1,
                                groups=Chead, bias=True)
            self.pw = nn.Conv2d(Chead, Chead, kernel_size=1, bias=True)

        self.act = nn.SiLU()
        self.tanh = nn.Tanh()


        self.alpha = nn.Parameter(torch.zeros(1))


        self.enable = True

    def forward(self, v):
        if not self.enable:
            return v

        Bwin, Nh, L, C = v.shape
        Wh, Ww = self.window_size
        assert L == Wh * Ww, f"L({L}) != Wh*Ww({Wh*Ww}) in VWindowSpatialGate"

        x = v.permute(0, 1, 3, 2).contiguous().view(Bwin * Nh, C, Wh, Ww)

        if self.use_asym:
            g = self.dw2(self.dw1(x))
            g = self.pw(self.act(g))
        else:
            g = self.pw(self.act(self.dw(x)))

        g = self.tanh(g)
        x = x * (1.0 + self.alpha * g)

        out = x.view(Bwin, Nh, C, L).permute(0, 1, 3, 2).contiguous()
        return out


class VChannelGate(nn.Module):


    def __init__(self, Chead, reduction=8):
        super().__init__()
        hidden = max(1, Chead // reduction)
        self.fc1 = nn.Linear(Chead, hidden, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, Chead, bias=True)
        self.tanh = nn.Tanh()

        self.alpha = nn.Parameter(torch.zeros(1))
        self.enable = True

    def forward(self, v):
        if not self.enable:
            return v


        Bwin, Nh, L, C = v.shape

        s = v.mean(dim=2)
        g = self.fc2(self.act(self.fc1(s)))
        g = self.tanh(g)
        g = g.unsqueeze(2)

        v_mod = v * (1.0 + self.alpha * g)
        return v_mod


class SCC(nn.Module):


    def __init__(self, dim, base_win_size, window_size, num_heads,
                 value_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads


        self.qv = DFE(dim, dim)
        self.proj = nn.Linear(dim, dim)


        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)


        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)


        head_dim = dim // (2 * num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(
            self.window_size[0] * self.window_size[1] //
            (self.base_win_size[0] * self.base_win_size[1]), 1
        )


        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)


        self.head_dim = head_dim
        self.v_gate_spatial = VWindowSpatialGate(self.head_dim, self.window_size, use_asym=True)
        self.v_gate_channel = VChannelGate(self.head_dim, reduction=8)

    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        H, W = self.window_size
        map_H, map_W = self.base_win_size

        x = x.view(
            B, num_h,
            map_H, H // map_H,
            map_W, W // map_W,
            C
        ).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(
            B, num_h, map_H * map_W, C, -1
        )
        x = self.spatial_linear(x).view(B, num_h, map_H * map_W, C)
        return x

    def spatial_self_correlation(self, q, v):
        B, num_head, L, C = q.shape


        v = self.spatial_linear_projection(v)


        corr_map = (q @ v.transpose(-2, -1)) / self.scale


        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)


        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.base_win_size[0],
            self.window_size[0] // self.base_win_size[0],
            self.base_win_size[1],
            self.window_size[1] // self.base_win_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(
            0, 1, 3, 5, 2, 4
        ).contiguous().view(
            self.window_size[0] * self.window_size[1],
            self.base_win_size[0] * self.base_win_size[1],
            self.num_heads,
            -1
        ).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        corr_map = corr_map + relative_position_bias.unsqueeze(0)


        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)

        return x

    def channel_self_correlation(self, q, v):
        B, num_head, L, C = q.shape


        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)


        corr_map = (q.transpose(-2, -1) @ v) / L


        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2, -1)).permute(0, 2, 1).contiguous().view(B, L, -1)

        return x

    def forward(self, x):


        xB, xH, xW, xC = x.shape
        qv = self.qv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)


        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0] * self.window_size[1], xC)


        B, L, C = qv.shape
        qv = qv.view(
            B, L, 2, self.num_heads, C // (2 * self.num_heads)
        ).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qv[0], qv[1]


        v_sp = self.v_gate_spatial(v)
        v_ch = self.v_gate_channel(v)


        x_spatial = self.spatial_self_correlation(q, v_sp)
        x_spatial = x_spatial.view(
            -1, self.window_size[0], self.window_size[1], C // 2
        )
        x_spatial = window_reverse(
            x_spatial, (self.window_size[0], self.window_size[1]), xH, xW
        )


        x_channel = self.channel_self_correlation(q, v_ch)
        x_channel = x_channel.view(
            -1, self.window_size[0], self.window_size[1], C // 2
        )
        x_channel = window_reverse(
            x_channel, (self.window_size[0], self.window_size[1]), xH, xW
        )


        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class HierarchicalTransformerBlock(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, base_win_size, window_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio


        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert window_size[0] % base_win_size[0] == 0, \
                "please ensure the window size is smaller than or divisible by the base window size"
            assert window_size[1] % base_win_size[1] == 0, \
                "please ensure the window size is smaller than or divisible by the base window size"

        self.norm1 = norm_layer(dim)
        self.correlation = SCC(
            dim, base_win_size=base_win_size, window_size=self.window_size,
            num_heads=num_heads, value_drop=value_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )

    def check_image_size(self, x, win_size):
        x = x.permute(0, 3, 1, 2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x, x_size, win_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)


        x = self.check_image_size(x, win_size)

        x = self.correlation(x)


        x = x[:, :H, :W, :].contiguous()


        x = x.view(B, H * W, C)
        x = self.norm1(x)


        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"mlp_ratio={self.mlp_ratio}")


class PatchMerging(nn.Module):


    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):


        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):


    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, hier_win_ratios=[1, 2, 4]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]


        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                base_win_size=base_win_size,
                window_size=(self.win_hs[i], self.win_ws[i]),
                mlp_ratio=mlp_ratio,
                drop=drop, value_drop=value_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])


        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        i = 0
        for blk in self.blocks:
            cur_win = (self.win_hs[i], self.win_ws[i])
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk, x, x_size, cur_win
                )
            else:
                x = blk(x, x_size, cur_win)
            i = i + 1

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"depth={self.depth}")


class PatchEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(
            B, self.embed_dim, x_size[0], x_size[1]
        )
        return x


class RHTB(nn.Module):


    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4,
                 resi_connection='1conv',
                 hier_win_ratios=[1, 2, 4]):
        super(RHTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mlp_ratio=mlp_ratio,
            drop=drop, value_drop=value_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':

            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim,
            norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim,
            norm_layer=None
        )

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(
                self.patch_unembed(self.residual_group(x, x_size), x_size)
            )
        ) + x


class Upsample(nn.Sequential):


    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f'scale {scale} is not supported. Supported scales: 2^n and 3.'
            )
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):


    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=bias
    )


class CALayer(nn.Module):


    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, mid, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class EPA(nn.Module):


    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.conv(x))
        return x * mask


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):


    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


class BMFANTLBlock(nn.Module):


    def __init__(self, dim, img_size=64, base_win_size=(8, 8),
                 depth=1, num_heads=6, mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 hier_win_ratios=[1, 2, 4]):
        super().__init__()
        self.dim = dim

        self.rhtb = RHTB(
            dim=dim,
            input_resolution=(img_size, img_size),
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            value_drop=value_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            img_size=img_size,
            patch_size=1,
            resi_connection='1conv',
            hier_win_ratios=hier_win_ratios
        )

    def forward(self, x):

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        out_tokens = self.rhtb(tokens, (H, W))
        out = out_tokens.transpose(1, 2).view(B, C, H, W)
        return out


class BMFANBlock(nn.Module):


    def __init__(self, channel, img_size=64, base_win_size=(8, 8),
                 depth=1, num_heads=6, mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 hier_win_ratios=[1, 2, 4]):
        super().__init__()

        base_win_size = tuple(base_win_size)

        if len(hier_win_ratios) == 0:
            small_ratio, mid_ratio, large_ratio = 1.0, 2.0, 4.0
        elif len(hier_win_ratios) == 1:
            small_ratio = mid_ratio = large_ratio = hier_win_ratios[0]
        elif len(hier_win_ratios) == 2:
            small_ratio = hier_win_ratios[0]
            mid_ratio = large_ratio = hier_win_ratios[1]
        else:
            small_ratio = hier_win_ratios[0]
            mid_ratio = hier_win_ratios[1]
            large_ratio = hier_win_ratios[2]

        self.tl1 = BMFANTLBlock(
            dim=channel,
            img_size=img_size,
            base_win_size=base_win_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            value_drop_rate=value_drop_rate,
            drop_path_rate=drop_path_rate,
            hier_win_ratios=[small_ratio],
        )
        self.tl2 = BMFANTLBlock(
            dim=channel,
            img_size=img_size,
            base_win_size=base_win_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            value_drop_rate=value_drop_rate,
            drop_path_rate=drop_path_rate,
            hier_win_ratios=[mid_ratio],
        )
        self.tl3 = BMFANTLBlock(
            dim=channel,
            img_size=img_size,
            base_win_size=base_win_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            value_drop_rate=value_drop_rate,
            drop_path_rate=drop_path_rate,
            hier_win_ratios=[large_ratio],
        )

        self.fusion = nn.Conv2d(channel * 3, channel, 1)
        self.ca = CoordAtt(channel, channel)

    def forward(self, x):
        shortcut = x

        y1 = self.tl1(x)
        y2 = self.tl2(y1)
        y3 = self.tl3(y2)

        out_fuse = self.fusion(torch.cat([y1, y2, y3], dim=1))
        out = self.ca(out_fuse)

        return out + shortcut


@ARCH_REGISTRY.register()
class BMFAN(nn.Module, PyTorchModelHubMixin):


    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 base_win_size=[8, 8], mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1.,
                 upsampler='pixelshuffledirect', resi_connection='1conv',
                 hier_win_ratios=[1, 2, 4],
                 **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        nf = embed_dim

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale
        self.upsampler = upsampler
        self.base_win_size = base_win_size


        self.conv_first = conv_layer(num_in_ch, nf, kernel_size=3)


        hit_depth = 1
        num_head = num_heads[0] if isinstance(num_heads, (list, tuple)) else num_heads

        self.fb1 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)
        self.fb2 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)
        self.fb3 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)
        self.fb4 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)
        self.fb5 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)
        self.fb6 = BMFANBlock(channel=nf, img_size=img_size, base_win_size=tuple(base_win_size),
                            depth=hit_depth, num_heads=num_head, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, value_drop_rate=value_drop_rate,
                            drop_path_rate=drop_path_rate, hier_win_ratios=hier_win_ratios)


        def make_fusion():
            return nn.Sequential(
                CALayer(channel=nf * 2, reduction=16),
                nn.Conv2d(nf * 2, nf, kernel_size=1, padding=0, bias=True),
                nn.GELU(),
                EPA(nf)
            )

        self.fusion1 = make_fusion()
        self.fusion2 = make_fusion()
        self.fusion3 = make_fusion()
        self.fusion4 = make_fusion()
        self.fusion5 = make_fusion()


        self.LR_conv = conv_layer(nf, nf, kernel_size=3)


        self.upsample = UpsampleOneStep(
            scale=upscale,
            num_feat=nf,
            num_out_ch=num_out_ch,
            input_resolution=(img_size, img_size)
        )

    def forward(self, x):
        H, W = x.shape[2:]


        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range


        y0 = self.conv_first(x)


        y1 = self.fb1(y0)
        y2 = self.fb2(y1)
        y3 = self.fb3(y2)
        y4 = self.fb4(y3)
        y5 = self.fb5(y4)
        y6 = self.fb6(y5)


        out1 = self.fusion1(torch.cat([y6, y5], dim=1))
        out2 = self.fusion2(torch.cat([out1, y4], dim=1))
        out3 = self.fusion3(torch.cat([out2, y3], dim=1))
        out4 = self.fusion4(torch.cat([out3, y2], dim=1))
        out5 = self.fusion5(torch.cat([out4, y1], dim=1))

        out_fused = y0 + out5
        out_lr = self.LR_conv(out_fused)


        out = self.upsample(out_lr)
        out = out / self.img_range + self.mean

        return out[:, :, :H * self.upscale, :W * self.upscale]


def make_model(args, parent=False):


    if isinstance(args.scale, (list, tuple)):
        upscale = int(args.scale[0])
    else:
        upscale = int(args.scale)

    raw_patch = getattr(args, 'patch_size', 64)

    if raw_patch % upscale == 0:
        img_size = raw_patch // upscale
    else:
        img_size = raw_patch

    n_colors = getattr(args, 'n_colors', 3)
    embed_dim = getattr(args, 'n_feats', 60)

    model = BMFAN(
        img_size=img_size,
        patch_size=1,
        in_chans=n_colors,
        embed_dim=embed_dim,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        base_win_size=[8, 8],
        mlp_ratio=2.,
        drop_rate=0.,
        value_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=upscale,
        img_range=1.,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
        hier_win_ratios=[1, 2, 4],
    )
    return model
