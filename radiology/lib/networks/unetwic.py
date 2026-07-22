import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock
from .blocks.test_block1 import UnetrUpCABlock
from .blocks.cbam import CBAM 
from .blocks.wtconv3d import WTConv3d

class UNETWIC(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=4,
            patch_size=2,
            kernel_size=7,
            exp_rate=4,
            feature_size=48,
            depths=[3, 3, 9, 3],
            drop_path_rate=0.0, 
            use_init_weights=False,
            is_conv_stem=False,
            skip_encoder_name=None,
            deep_sup=False,
            first_feature_size_half=False,
            wave_level=1, 
            **kwargs,
    ) -> None:
        super().__init__()
        
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        
        if first_feature_size_half:
            first_feature_size = feature_sizes[0] // 2
        else:
            first_feature_size = feature_sizes[0]
        
        decoder_norm_name = 'instance'
        res_block = True
        spatial_dims = 3
        
        # --- Encoder 0 (Stem) ---
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=first_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        
        self.backbone = HybridBackbone(
            in_channels=in_channels,
            patch_size=patch_size,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            feature_sizes=feature_sizes,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_init_weights=use_init_weights,
            is_conv_stem=is_conv_stem,
            wave_level=wave_level
        )

         # Skip Connections
        self.skip_encoder_name = skip_encoder_name
        if self.skip_encoder_name == 'cbam': 
             self.skip_encoder0 = nn.Identity()
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
             self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)
        
        elif self.skip_encoder_name == 'hybrid': 
             # Level 0 (Stem)
             self.skip_encoder0 = nn.Identity() 
             # Level 1 (Stage 0): CBAM 
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7) 
             # Level 2 (Stage 1): CBAM 
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             # Level 3 (Stage 2): Identity 
             self.skip_encoder3 = nn.Identity()
             # Level 4 (Stage 3): Identity
             self.skip_encoder4 = nn.Identity()

        self.bottleneck = nn.Sequential(
            LayerNorm(feature_sizes[3], eps=1e-6, data_format="channels_first"),
            nn.Conv3d(feature_sizes[3], feature_sizes[3]*2, kernel_size=2, stride=2),
            CBAM(feature_sizes[3]*2, reduction=16, kernel_size=7)
        )

        # Decoder 5  
        self.decoder5 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2, 
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2, 
            norm_name=decoder_norm_name,
            res_block=res_block
        )
             
        # Level 4
        self.decoder4 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )
        #self.wt_dec4 = WTConv3d(feature_sizes[2], feature_sizes[2], wt_levels=wave_level)

        # Level 3
        self.decoder3 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        #self.wt_dec3 = WTConv3d(feature_sizes[1], feature_sizes[1], wt_levels=wave_level)

        # Level 2
        self.decoder2 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        #self.wt_dec2 = WTConv3d(feature_sizes[0], feature_sizes[0], wt_levels=wave_level)
        
        # Level 1
        self.decoder1 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels)
        
        # Deep Supervision
        self.deep_sup = deep_sup
        if deep_sup:
            self.ds_block1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[0], out_channels=out_channels)
            self.ds_block2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[1], out_channels=out_channels)

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        hidden_states_out = self.backbone(x) 
        enc1, enc2, enc3, enc4 = hidden_states_out

        # Skip Connection Refinement
        if self.skip_encoder_name == 'cbam':
            enc1 = self.skip_encoder1(enc1)
            enc2 = self.skip_encoder2(enc2)
            enc3 = self.skip_encoder3(enc3)
            enc4 = self.skip_encoder4(enc4)
        elif self.skip_encoder_name == 'hybrid':
            enc1 = enc1 + self.skip_encoder1(enc1) 
            enc2 = enc2 + self.skip_encoder2(enc2)
            # Level 3 & 4: Identity
            enc3 = self.skip_encoder3(enc3)
            enc4 = self.skip_encoder4(enc4)

        # Bottleneck       
        bn = self.bottleneck(enc4)

        # Decoder
        dec5 = self.decoder5(bn, enc4)
        
        dec4 = self.decoder4(dec5, enc3)
        #dec4 = dec4 + self.wt_dec4(dec4)
        dec3 = self.decoder3(dec4, enc2)
        #dec3 = dec3 + self.wt_dec3(dec3) 

        dec2 = self.decoder2(dec3, enc1)
        #dec2 = dec2 + self.wt_dec2(dec2) 

        dec1 = self.decoder1(dec2, enc0)
        out = self.out_block(dec1)
        
        if self.deep_sup and self.training:
            out1 = self.ds_block1(dec2)
            out2 = self.ds_block2(dec3)
            return [out, out1, out2]
        else:
            return out

# Hybrid Backbone
class HybridBackbone(nn.Module):
    def __init__(self, in_channels, patch_size, kernel_size, exp_rate, feature_sizes, depths, drop_path_rate, use_init_weights, is_conv_stem, wave_level):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        
        # Stem
        if is_conv_stem:
            stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=7, stride=patch_size, padding=3),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        else:
             stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(feature_sizes[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(feature_sizes[i], feature_sizes[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            #curr_wave_level = wave_level if i < 2 else 1
            #curr_wave_level = 1 if i < 2 else wave_level
            stage = nn.Sequential(
                *[
                    InceptionWTBlock(  
                        dim=feature_sizes[i], 
                        kernel_size=kernel_size,
                        exp_rate=exp_rate,
                        drop_path=dp_rates[cur + j],
                        wt_levels=wave_level 
                    )
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        if use_init_weights:
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class InceptionWTBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        kernel_size=11, 
        exp_rate=4, 
        drop_path=0., 
        wt_levels=1, 
        wt_type='db1'
    ):
        super().__init__()
        self.gc = int(dim * 0.2)
        #self.alpha = nn.Parameter(torch.tensor(0.5))
        # 空間分支：4 路卷積
        self.dwconv_hwd = nn.Conv3d(self.gc, self.gc, kernel_size=5, padding=2, groups=self.gc)
        self.dwconv_h = nn.Conv3d(self.gc, self.gc, kernel_size=(kernel_size, kernel_size, 1), 
                                   padding=(kernel_size//2, kernel_size//2, 0), groups=self.gc)
        self.dwconv_w = nn.Conv3d(self.gc, self.gc, kernel_size=(1, kernel_size, kernel_size), 
                                   padding=(0, kernel_size//2, kernel_size//2), groups=self.gc)
        self.dwconv_d = nn.Conv3d(self.gc, self.gc, kernel_size=(kernel_size, 1, kernel_size), 
                                   padding=(kernel_size//2, 0, kernel_size//2), groups=self.gc)
        
        # 定義切分索引
        self.split_indexes = (dim - 4 * self.gc, self.gc, self.gc, self.gc, self.gc)

        inter_dim = max(8, dim // 4) 
        self.freq_path = nn.Sequential(
            nn.Conv3d(dim, inter_dim, 1),
            WTConv3d(inter_dim, inter_dim, kernel_size=5, wt_levels=wt_levels, wt_type=wt_type),
            nn.Conv3d(inter_dim, dim, 1),
            nn.Sigmoid() 
        )
        #self.beta = nn.Parameter(torch.ones(dim, 1, 1, 1) * 0.1 )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, exp_rate * dim)
        self.act = nn.GELU()
        self.grn = GRN(exp_rate * dim)
        self.pwconv2 = nn.Linear(exp_rate * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x_id, x_hwd, x_w, x_h, x_d = torch.split(x, self.split_indexes, dim=1) 
        
        x_spatial = torch.cat((
            x_id,
            self.dwconv_hwd(x_hwd), 
            self.dwconv_w(x_w), 
            self.dwconv_h(x_h), 
            self.dwconv_d(x_d)
        ), dim=1) 
        
        freq_attn = self.freq_path(x)
        x_modulated = x_spatial  * (1 + freq_attn)
        
        x = x_modulated.permute(0, 2, 3, 4, 1) # Channel Last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # Channel First

        x = input + self.drop_path(x)
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x