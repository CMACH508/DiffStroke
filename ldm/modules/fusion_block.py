import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.encoders.adapter import ResnetBlock
from adaIN import adain
v1 = True

# v1
if v1:
    class fusionnet(nn.Module):
        def __init__(self, ch=[320+16, 640, 1280, 1280]):
            super().__init__()
            self.layers_sa = nn.ModuleList([nn.Sequential(nn.TransformerEncoderLayer(ch[i],8,1024,activation=nn.SiLU(),batch_first=True,norm_first=True),
                                        nn.TransformerEncoderLayer(ch[i],8,1024,activation=nn.SiLU(),batch_first=True,norm_first=True),
                                        nn.TransformerEncoderLayer(ch[i],8,1024,activation=nn.SiLU(),batch_first=True,norm_first=True)) 
                                        for i in range(len(ch))])
            self.mask_para = nn.Parameter(torch.randn(1,64*64,16))
            self.linear = nn.Conv2d(16,1,3,1,1)

        def forward(self, adapter_feature, ori_img_feature):
            '''
            adapter_feature: Sketch features : [bs,320,64,64], [bs,640,32,32], [bs,1280,16,16], [bs,1280,8,8]
            ori_img_feature: ori_image + noise feature: [bs,320,64,64], [bs,640,32,32], [bs,1280,16,16], [bs,1280,8,8]
            return:
            [h_edge, mask]
            '''
            hs = []
            for i, (layer_sa) in enumerate(self.layers_sa):
                b1,c1,h1,w1 = adapter_feature[i].shape
                b2,c2,h2,w2 = ori_img_feature[i].shape
                assert c1==c2, "The dim c of Adapter_fearture mismatches ori_image_feature"
                ad_f = adapter_feature[i].view(b1,c1,h1*w1).permute(0,2,1)
                ori_f = ori_img_feature[i].view(b2,c2,h2*w2).permute(0,2,1)
                if i==0:
                    tmp = torch.cat([ori_f+ad_f, self.mask_para.repeat(b1,1,1)], dim=-1)
                    c1,c2 = c1+16,c2+16
                else:
                    tmp = ori_f+ad_f
                h_edge = layer_sa(tmp)
                h_edge = h_edge.permute(0,2,1).view(b2,c2,h2,w2)

                if i==0:
                    h_mask = torch.sigmoid(self.linear(h_edge[:,320:,:,:]))
                    h_edge = h_edge[:,:320,:,:]
                #h_edge =1.0 *h_edge + 0.*adapter_feature[i]#adain(adapter_feature[i], h_edge) 

                hs.append([h_edge,h_mask])
                #hs.append([h_edge,torch.ones_like(h_mask).cuda()])
                #hs.append([adapter_feature[i],h_mask])
                #hs.append([adapter_feature[i],torch.ones_like(h_mask).cuda()])
            return hs
else:
    class fusionnet(nn.Module):
        def __init__(self, ch=[320, 640, 1280, 1280]):
            super().__init__()
            # 所有层的输入通道数都需要增加16
            ch_with_mask = [c + 16 for c in ch]
            self.layers_sa = nn.ModuleList([nn.Sequential(
                nn.TransformerEncoderLayer(ch_with_mask[i], 8, 1024, activation=nn.SiLU(), batch_first=True, norm_first=True),
                nn.TransformerEncoderLayer(ch_with_mask[i], 8, 1024, activation=nn.SiLU(), batch_first=True, norm_first=True),
                nn.TransformerEncoderLayer(ch_with_mask[i], 8, 1024, activation=nn.SiLU(), batch_first=True, norm_first=True)) 
                for i in range(len(ch_with_mask))])
            
            # 为每一层创建不同的16通道可学习参数
            self.mask_paras = nn.ParameterList([
                nn.Parameter(torch.randn(1, 64 * 64, 16)),    # 第0层: 64×64
                nn.Parameter(torch.randn(1, 32 * 32, 16)),    # 第1层: 32×32  
                nn.Parameter(torch.randn(1, 16 * 16, 16)),    # 第2层: 16×16
                nn.Parameter(torch.randn(1, 8 * 8, 16))       # 第3层: 8×8
            ])
            
            # 为每一层创建不同的卷积层来生成掩膜
            self.mask_convs = nn.ModuleList([
                nn.Conv2d(16, 1, 3, 1, 1),  # 第0层
                nn.Conv2d(16, 1, 3, 1, 1),  # 第1层
                nn.Conv2d(16, 1, 3, 1, 1),  # 第2层
                nn.Conv2d(16, 1, 3, 1, 1)   # 第3层
            ])

        def forward(self, adapter_feature, ori_img_feature):
            '''
            adapter_feature: Sketch features : [bs,320,64,64], [bs,640,32,32], [bs,1280,16,16], [bs,1280,8,8]
            ori_img_feature: ori_image + noise feature: [bs,320,64,64], [bs,640,32,32], [bs,1280,16,16], [bs,1280,8,8]
            return:
            [h_edge, mask]
            '''
            hs = []
            mask_list = []  # 存储每一层的掩膜
            
            for i, (layer_sa, mask_para, mask_conv) in enumerate(zip(self.layers_sa, self.mask_paras, self.mask_convs)):
                b1, c1, h1, w1 = adapter_feature[i].shape
                b2, c2, h2, w2 = ori_img_feature[i].shape
                assert c1 == c2, "The dim c of Adapter_fearture mismatches ori_image_feature"
                
                ad_f = adapter_feature[i].view(b1, c1, h1*w1).permute(0, 2, 1)
                ori_f = ori_img_feature[i].view(b2, c2, h2*w2).permute(0, 2, 1)
                
                # 每一层都拼接对应的可学习参数
                tmp = torch.cat([ori_f + ad_f, mask_para.repeat(b1, 1, 1)], dim=-1)
                
                h_edge = layer_sa(tmp)
                h_edge = h_edge.permute(0, 2, 1).view(b2, c2+16, h2, w2)  # 注意维度变化
                
                # 分离出特征和掩膜部分
                h_edge_feature = h_edge[:, :c2, :, :]  # 原始特征部分
                h_mask_feature = h_edge[:, c2:, :, :]  # 掩膜特征部分 (16通道)
                
                # 生成当前层的掩膜
                h_mask_layer = torch.sigmoid(mask_conv(h_mask_feature))  # [bs, 1, h2, w2]
                
                # 将当前层掩膜上采样到64×64分辨率
                h_mask_upsampled = F.interpolate(h_mask_layer, size=(64, 64), mode='bilinear', align_corners=False)
                mask_list.append(h_mask_upsampled)
                
                # 暂时保存当前层的特征和掩膜
                hs.append([h_edge_feature, h_mask_layer])
            
            # 将所有层的掩膜相加并取平均
            h_mask_final = torch.stack(mask_list, dim=0).mean(dim=0)  # [bs, 1, 64, 64]
            
            # 更新所有层的掩膜输出为最终的融合掩膜
            for i in range(len(hs)):
                hs[i][1] = h_mask_final
            return hs