import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD


class CrossScaleFeatureFusion(nn.Module):
    """
    Feature-level multi-scale fusion via channel attention.
    
    Projects all decoder features to the same resolution (d1's H/4 × W/4)
    and channel dimension, then uses squeeze-excitation channel attention
    to learn which scale features are most useful at each channel.
    
    Unlike prediction-level fusion (which operates on 4 scalar values per pixel),
    this operates on rich multi-channel features (~256 channels), giving the
    attention module much more information to make informed scale selections.
    """
    def __init__(self, channels, target_ch=64, se_ratio=16):
        """
        Args:
            channels: list of decoder channel dims [ch4, ch3, ch2, ch1]
                      e.g. [512, 320, 128, 64] for pvt_v2_b2
            target_ch: channel dim to project each scale to (default: 64)
            se_ratio: reduction ratio for SE attention (default: 16)
        """
        super(CrossScaleFeatureFusion, self).__init__()
        self.num_scales = len(channels)
        concat_ch = target_ch * self.num_scales  # 256 for 4 scales

        # 1×1 projection for each decoder scale → target_ch
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, target_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_ch),
                nn.ReLU(inplace=True)
            ) for ch in channels
        ])

        # Squeeze-Excitation channel attention on concatenated features
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(concat_ch, concat_ch // se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(concat_ch // se_ratio, concat_ch),
            nn.Sigmoid()
        )

        # Final reduction: concat_ch → target_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(concat_ch, target_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec_outs):
        """
        Args:
            dec_outs: list of decoder feature maps [d4, d3, d2, d1]
                      d4: [B, 512, H/32, W/32], d1: [B, 64, H/4, W/4]
        Returns:
            fused: [B, target_ch, H/4, W/4] fused feature map
        """
        target_size = dec_outs[-1].shape[2:]  # d1's spatial size (H/4, W/4)

        projected = []
        for i, (feat, proj) in enumerate(zip(dec_outs, self.projections)):
            f = proj(feat)  # project to target_ch
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            projected.append(f)

        concat = torch.cat(projected, dim=1)  # [B, 256, H/4, W/4]

        # Channel attention
        attn = self.se(concat)  # [B, 256]
        attn = attn.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        concat = concat * attn  # channel-wise reweighting

        fused = self.reduce(concat)  # [B, 64, H/4, W/4]
        return fused


class EMCADNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
        super(EMCADNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = pretrained_dir + '/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = pretrained_dir + '/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = pretrained_dir + '/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = pretrained_dir + '/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = pretrained_dir + '/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
        # Cross-scale feature fusion module
        self.feature_fusion = CrossScaleFeatureFusion(channels=channels, target_ch=channels[3])
        self.out_head_fused = nn.Conv2d(channels[3], num_classes, 1)
        
        print('Model %s created, param count: %d' %
                     ('Cross-Scale Feature Fusion: ', sum([m.numel() for m in self.feature_fusion.parameters()]) + sum([m.numel() for m in self.out_head_fused.parameters()])))
        
    def forward(self, x, mode='test'):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # encoder
        x1, x2, x3, x4 = self.backbone(x)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        # Cross-scale feature fusion → fused prediction
        fused_feat = self.feature_fusion(dec_outs)  # [B, 64, H/4, W/4]
        p_fused = self.out_head_fused(fused_feat)    # [B, 1, H/4, W/4]
        p_fused = F.interpolate(p_fused, scale_factor=4, mode='bilinear')

        return [p4, p3, p2, p1, p_fused]
               

        
if __name__ == '__main__':
    model = EMCADNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())

