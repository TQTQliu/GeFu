import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .feature_net import FeatureNet
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .nerf import NeRF
from .feature_net import AutoEncoder
from .utils import *
import imageio
import os

def convgnrelu(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, group_channel=8):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels),
        nn.ReLU(inplace=True)
    )
    

class InterViewAAModule(nn.Module):
    def __init__(self,in_channels=32, bias=True):
        super(InterViewAAModule, self).__init__()
        self.reweight_network = nn.Sequential(
                                    convgnrelu(in_channels, 4, kernel_size=3, stride=1, dilation=1, bias=bias),
                                    resnet_block_gn(4, kernel_size=1),
                                    nn.Conv2d(4, 1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                )
    
    def forward(self, x):
        return self.reweight_network(x)
def resnet_block_gn(in_channels,  kernel_size=3, dilation=[1,1], bias=True, group_channel=8):
    return ResnetBlockGn(in_channels, kernel_size, dilation, bias=bias, group_channel=group_channel)

class ResnetBlockGn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, group_channel=8):
        super(ResnetBlockGn, self).__init__()
        self.stem = nn.Sequential(
            convgnrelu(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias, group_channel=group_channel), 
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.GroupNorm(int(max(1, in_channels / group_channel)), in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.stem(x) + x
        out = self.relu(out)
        return out
    
class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.gefu.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i)))*2)
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i)))*2)
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            nerf_l = NeRF(feat_ch=cfg.gefu.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}', nerf_l)
            
        h=64
        self.autoencoder = AutoEncoder(91, h, h)
        self.conv_transmit = AutoEncoder(91, 16, 16)
        self.color_rf = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 3),
                nn.Sigmoid())
        self.omega = InterViewAAModule(16)
        self.out_rf = nn.Sequential(
                nn.Linear(h+3, 8),
                nn.ReLU())
        self.out_nf = nn.Sequential(
                nn.Linear(91, 8),
                nn.ReLU())
        self.wgt_rf = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid())
        self.wgt_nf = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid())
        
    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        size =  kwargs['size']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.gefu.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.gefu.cas_config.render_scale[level])
        up_feat_scale = cfg.gefu.cas_config.render_scale[level] / cfg.gefu.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.gefu.cas_config.render_scale[level]), int(W_O * cfg.gefu.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
        net_output, fea_transmit = nerf_model(vox_feat, img_feat_rgb_dir, size)
        net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
        outputs = utils.raw2outputs(net_output, z_vals, cfg.gefu.white_bkgd, fea_transmit)
        return outputs

    def batchify_rays(self, rays, **kwargs):
        all_ret = {}
        chunk = cfg.gefu.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:, i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret


    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward_render(self, ret, batch):
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)


    def forward(self, batch):
        B, _, _, H_img, W_img = batch['src_inps'].shape
        pred_rgb_nb_list = []
        if not cfg.save_video:
            feats = self.forward_feat(batch['src_inps'])
            ret = {}
            depth, std, near_far = None, None, None
            tgt_fea = None
            for i in range(cfg.gefu.cas_config.num):
                H, W = int(H_img*cfg.gefu.cas_config.render_scale[i]), int(W_img*cfg.gefu.cas_config.render_scale[i])
                feature_volume, depth_values, near_far = utils.build_feature_volume(
                        feats[f'level_{i}'],
                        batch,
                        D=cfg.gefu.cas_config.volume_planes[i],
                        depth=depth,
                        std=std,
                        near_far=near_far,
                        level=i,
                        tgt_fea=tgt_fea,
                        omega_net = self.omega)
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                if not cfg.gefu.cas_config.render_if[i]:
                    continue
                rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                im_feat_level = cfg.gefu.cas_config.render_im_feat_level[i]
                ret_i = self.batchify_rays(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'],
                        nerf_model=getattr(self, f'nerf_{i}'),
                        level=i,
                        size=(H,W)
                        )
                if cfg.gefu.cas_config.depth_inv[i]:
                    ret_i.update({'depth_mvs': 1./depth})
                else:
                    ret_i.update({'depth_mvs': depth})
                ret_i.update({'std': std})
                if ret_i['rgb'].isnan().any():
                    __import__('ipdb').set_trace()
                ret.update({key+f'_level{i}': ret_i[key] for key in ret_i if key != 'fea_transmit'})
                if i < cfg.gefu.cas_config.num-1:
                    tgt_rgb = ret_i['rgb'].reshape(B,H,W,3)
                    tgt_rgb = tgt_rgb.permute(0,3,1,2)
                    fea_transmit = ret_i['fea_transmit']
                    fea_transmit = fea_transmit.reshape(B,H,W,fea_transmit.shape[-1]).permute(0,3,1,2)
                    fea_transmit = torch.cat((fea_transmit,tgt_rgb),dim=1)
                    tgt_fea = self.conv_transmit(fea_transmit)
            fea_transmit = ret_i['fea_transmit']
            fea_transmit = fea_transmit.reshape(B,H,W,fea_transmit.shape[-1])
            tgt_rgb = ret_i['rgb'].reshape(B,H,W,3).clone()
            fea_nf = torch.cat((fea_transmit,tgt_rgb),dim=-1)
            fea_rf = self.autoencoder(fea_nf.permute(0,3,1,2)).flatten(2).permute(0,2,1)
            rgb_rf = self.color_rf(fea_rf)
            fea_rf = torch.cat((fea_rf,rgb_rf), dim=-1)
            fea_rf = self.out_rf(fea_rf)
            fea_rf = fea_rf.reshape(B,H,W,fea_rf.shape[-1]).permute(0,3,1,2)
            fea_nf = self.out_nf(fea_nf)
            fea_nf = fea_nf.reshape(B,H,W,fea_nf.shape[-1]).permute(0,3,1,2)

            depth = ret[f'depth_level{cfg.gefu.cas_config.num-1}'].reshape(B,H,W)
            feature_src = feats['level_2']
            S = feature_src.shape[1]
            proj_mats = get_proj_mats(batch, src_scale=1, tar_scale=1)
            depth_values = depth.unsqueeze(1)
            volume_adapt_rf, volume_adapt_nf = None, None
            for s in range(S):
                feature_s = feature_src[:, s]
                proj_mat = proj_mats[:, s]
                warped_volume, _ = homo_warp(feature_s, proj_mat, depth_values)
                warped_volume = warped_volume.squeeze(2)
                cost_volume_rf = (fea_rf - warped_volume).pow_(2)
                cost_volume_nf = (fea_nf - warped_volume).pow_(2)
                if volume_adapt_rf is None:
                    volume_adapt_rf = cost_volume_rf
                    volume_adapt_nf = cost_volume_nf
                else:
                    volume_adapt_rf = volume_adapt_rf + cost_volume_rf
                    volume_adapt_nf = volume_adapt_nf + cost_volume_nf
                del warped_volume
        
            volume_adapt_rf = volume_adapt_rf.flatten(2).permute(0,2,1)
            volume_adapt_nf = volume_adapt_nf.flatten(2).permute(0,2,1)

            w_rf = self.wgt_rf(volume_adapt_rf)
            w_nf = self.wgt_nf(volume_adapt_nf)
            w = torch.cat((w_nf,w_rf),dim=-1)
            w = F.softmax(w,dim=-1)
            w_nf = w[...,0].unsqueeze(-1)
            w_rf = w[...,1].unsqueeze(-1)
            rgb_nf = ret[f'rgb_level{cfg.gefu.cas_config.num-1}']
            ret[f'rgb_b_level{cfg.gefu.cas_config.num-1}'] =  rgb_nf.clone()
            ret[f'rgb_r_level{cfg.gefu.cas_config.num-1}'] =  rgb_rf.clone()
            ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] =  w_nf * rgb_nf  + w_rf  * rgb_rf
            if cfg.gefu.reweighting:
                ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] = (ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] \
                + ret[f'rgb_b_level{cfg.gefu.cas_config.num-1}']) / 2
            ret['w_rf'] = w_rf
            ret['w_nf'] = w_nf
            return ret
        else:
            for _, meta in enumerate(batch['rendering_video_meta']):
                batch['tar_ext'][:,:3] = meta['tar_ext'][:,:3]
                batch['rays_0'] = meta['rays_0']
                batch['rays_1'] = meta['rays_1']
                feats = self.forward_feat(batch['src_inps'])
                ret = {}
                depth, std, near_far = None, None, None
                tgt_fea = None
                for i in range(cfg.gefu.cas_config.num):
                    H, W = int(H_img*cfg.gefu.cas_config.render_scale[i]), int(W_img*cfg.gefu.cas_config.render_scale[i])
                    feature_volume, depth_values, near_far = utils.build_feature_volume(
                            feats[f'level_{i}'],
                            batch,
                            D=cfg.gefu.cas_config.volume_planes[i],
                            depth=depth,
                            std=std,
                            near_far=near_far,
                            level=i,
                            tgt_fea=tgt_fea,
                            omega_net = self.omega)
                    feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                    depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                    if not cfg.gefu.cas_config.render_if[i]:
                        continue
                    rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                    im_feat_level = cfg.gefu.cas_config.render_im_feat_level[i]
                    ret_i = self.batchify_rays(
                            rays=rays,
                            feature_volume=feature_volume,
                            batch=batch,
                            im_feat=feats[f'level_{im_feat_level}'],
                            nerf_model=getattr(self, f'nerf_{i}'),
                            level=i,
                            size=(H,W)
                            )
                    if cfg.gefu.cas_config.depth_inv[i]:
                        ret_i.update({'depth_mvs': 1./depth})
                    else:
                        ret_i.update({'depth_mvs': depth})
                    ret_i.update({'std': std})
                    if ret_i['rgb'].isnan().any():
                        __import__('ipdb').set_trace()
                    ret.update({key+f'_level{i}': ret_i[key] for key in ret_i if key != 'fea_transmit'})
                    if i < cfg.gefu.cas_config.num-1:
                        tgt_rgb = ret_i['rgb'].reshape(B,H,W,3)
                        tgt_rgb = tgt_rgb.permute(0,3,1,2)
                        fea_transmit = ret_i['fea_transmit']
                        fea_transmit = fea_transmit.reshape(B,H,W,fea_transmit.shape[-1]).permute(0,3,1,2)
                        fea_transmit = torch.cat((fea_transmit,tgt_rgb),dim=1)
                        tgt_fea = self.conv_transmit(fea_transmit)
                fea_transmit = ret_i['fea_transmit']
                fea_transmit = fea_transmit.reshape(B,H,W,fea_transmit.shape[-1])
                tgt_rgb = ret_i['rgb'].reshape(B,H,W,3).clone()
                fea_nf = torch.cat((fea_transmit,tgt_rgb),dim=-1)
                fea_rf = self.autoencoder(fea_nf.permute(0,3,1,2)).flatten(2).permute(0,2,1)
                rgb_rf = self.color_rf(fea_rf)
                fea_rf = torch.cat((fea_rf,rgb_rf), dim=-1)
                fea_rf = self.out_rf(fea_rf)
                fea_rf = fea_rf.reshape(B,H,W,fea_rf.shape[-1]).permute(0,3,1,2)
                fea_nf = self.out_nf(fea_nf)
                fea_nf = fea_nf.reshape(B,H,W,fea_nf.shape[-1]).permute(0,3,1,2)

                depth = ret[f'depth_level{cfg.gefu.cas_config.num-1}'].reshape(B,H,W)
                feature_src = feats['level_2']
                S = feature_src.shape[1]
                proj_mats = get_proj_mats(batch, src_scale=1, tar_scale=1)
                depth_values = depth.unsqueeze(1)
                volume_adapt_rf, volume_adapt_nf = None, None
                for s in range(S):
                    feature_s = feature_src[:, s]
                    proj_mat = proj_mats[:, s]
                    warped_volume, _ = homo_warp(feature_s, proj_mat, depth_values)
                    warped_volume = warped_volume.squeeze(2)
                    cost_volume_rf = (fea_rf - warped_volume).pow_(2)
                    cost_volume_nf = (fea_nf - warped_volume).pow_(2)
                    if volume_adapt_rf is None:
                        volume_adapt_rf = cost_volume_rf
                        volume_adapt_nf = cost_volume_nf
                    else:
                        volume_adapt_rf = volume_adapt_rf + cost_volume_rf
                        volume_adapt_nf = volume_adapt_nf + cost_volume_nf
                    del warped_volume
            
                volume_adapt_rf = volume_adapt_rf.flatten(2).permute(0,2,1)
                volume_adapt_nf = volume_adapt_nf.flatten(2).permute(0,2,1)

                w_rf = self.wgt_rf(volume_adapt_rf)
                w_nf = self.wgt_nf(volume_adapt_nf)
                w = torch.cat((w_nf,w_rf),dim=-1)
                w = F.softmax(w,dim=-1)
                w_nf = w[...,0].unsqueeze(-1)
                w_rf = w[...,1].unsqueeze(-1)
                rgb_nf = ret[f'rgb_level{cfg.gefu.cas_config.num-1}']
                ret[f'rgb_b_level{cfg.gefu.cas_config.num-1}'] =  rgb_nf.clone()
                ret[f'rgb_r_level{cfg.gefu.cas_config.num-1}'] =  rgb_rf.clone()
                ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] =  w_nf * rgb_nf  + w_rf  * rgb_rf
                if cfg.gefu.reweighting:
                    ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] = (ret[f'rgb_level{cfg.gefu.cas_config.num-1}'] \
                    + ret[f'rgb_b_level{cfg.gefu.cas_config.num-1}']) / 2
                b=0
                video_path = os.path.join(cfg.result_dir, '{}_{}_{}.mp4'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                render_novel = ret[f'rgb_level{cfg.gefu.cas_config.num-1}'].reshape(H_img,W_img,3)
                if cfg.gefu.eval_center:
                    H_crop, W_crop = int(H_img*0.1), int(W_img*0.1)
                    render_novel = render_novel[H_crop:-H_crop, W_crop:-W_crop,:]
                pred_rgb_nb_list.append((render_novel.data.cpu().numpy()*255).astype(np.uint8))
            imageio.mimwrite(video_path, np.stack(pred_rgb_nb_list), fps=10, quality=10)
    
