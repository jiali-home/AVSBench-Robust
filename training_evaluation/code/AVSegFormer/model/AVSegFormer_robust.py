import torch
import torch.nn as nn
from .backbone import build_backbone
# from .neck import build_neck
from .head import build_head
from .vggish import VGGish
from typing import Optional, List, Union, Literal
import torch.nn.functional as F


class AVSegFormer_robust(nn.Module):
    def __init__(self,
                 backbone,
                 vggish,
                 head,
                 neck=None,
                 audio_dim=128,
                 embed_dim=256,
                 T=5,
                 freeze_audio_backbone=True,
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.T = T
        self.freeze_audio_backbone = freeze_audio_backbone
        self.backbone = build_backbone(**backbone)
        self.vggish = VGGish(**vggish)
        self.head = build_head(**head)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

        if self.freeze_audio_backbone:
            for p in self.vggish.parameters():
                p.requires_grad = False
        self.freeze_backbone(True)

        self.neck = neck
        # if neck is not None:
        #     self.neck = build_neck(**neck)
        # else:
        #     self.neck = None

        self.cosine_sim = CosineModule()

    def freeze_backbone(self, freeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag

            return out

    def extract_feat(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats


    def forward(self, audio, frames, vid_temporal_mask_flag=None, gumbel=False):
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)
        with torch.no_grad():
            audio_feat = self.vggish(audio)  # [B*T,128]

        audio_feat = audio_feat.unsqueeze(1)
        audio_feat = self.audio_proj(audio_feat)
        img_feat = self.extract_feat(frames)
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)

        pred, mask_feature = self.head(img_feat, audio_feat)
        pred = self.mul_temporal_mask(pred, vid_temporal_mask_flag)
        mask_feature = self.mul_temporal_mask(
            mask_feature, vid_temporal_mask_flag)
        
        similarity = self.cosine_sim(audio_feat, img_feat[3])

        return pred, mask_feature, similarity


class CosineModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.audio_proj = nn.Linear(256, 512)  # 将音频特征投影到与视觉特征相同的通道维度
        
    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        # audio_feat: [5, 1, 256]
        # visual_feat: [5, 512, 7, 7]
        
        audio_feat = audio_feat.squeeze(1)  # [5, 256]
        audio_feat = self.audio_proj(audio_feat)  # [5, 512]
        
        visual_feat = F.adaptive_avg_pool2d(visual_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [5, 512]
        similarity = self.cosine_similarity(audio_feat, visual_feat)  # [5]
        
        return similarity