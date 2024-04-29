import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .cost_reg_net import SigCostRegNet
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_model_k, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_k, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_k, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        return q, attn

class NeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(NeRF, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(8+16, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
                nn.Linear(64+24+feat_ch+4, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, 1),
                nn.ReLU())
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)
        self.regnet = SigCostRegNet(hid_n)
        self.attn = MultiHeadAttention(8,hid_n,feat_ch+4,4,4)
        

    def forward(self, vox_feat, img_feat_rgb_dir, size):
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2]
        img_feat = self.agg(img_feat_rgb_dir)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)
        x = self.lr0(vox_img_feat)
        x = x.reshape(B,H,W,-1,x.shape[-1])
        x = x.permute(0,4,3,1,2)
        x = self.regnet(x)
        x = x.permute(0,1,3,4,2).flatten(2).permute(0,2,1)
        q = x.squeeze(0).unsqueeze(1)
        k = img_feat_rgb_dir.squeeze(0)
        x,_ = self.attn(q,k,k)
        x = x.squeeze(1).unsqueeze(0)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = self.sigma(x)
        x = torch.cat((x, vox_img_feat), dim=-1)
        fea_transmit = x.clone()
        x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x), dim=-2)
        color = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        return torch.cat([color, sigma], dim=-1), fea_transmit

class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.gefu.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.gefu.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)

class MVSNeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(MVSNeRF, self).__init__()
        self.hid_n = hid_n
        self.lr0 = nn.Sequential(nn.Linear(8+feat_ch*3, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
                nn.Linear(hid_n, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, 3))
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir):
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        # img_feat = self.agg(img_feat_rgb_dir)
        img_feat = torch.cat([img_feat_rgb_dir[..., i, :-4] for i in range(N_views)] , dim=-1)
        S = img_feat_rgb_dir.shape[2]
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)
        x = self.lr0(vox_img_feat)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = self.sigma(x)
        # x = torch.cat((x, vox_img_feat), dim=-1)
        # x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        # x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color = torch.sigmoid(self.color(x))
        return torch.cat([color, sigma], dim=-1)



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

