import torch
import torch.nn as nn
from transformers import BertModel
from config import *

def load_resnet50():
    resnet = torch.hub.load('pytorch/vision:v0.10.0', IMAGE_MODEL_NAME, weights="ResNet50_Weights.DEFAULT")
    resnet.fc = nn.Identity() 
    return resnet

class LateFusionModel(nn.Module):
    def __init__(self, dropout_rate=BASE_DROPOUT_RATE):
        super(LateFusionModel, self).__init__()

        self.text_encoder = BertModel.from_pretrained(TEXT_MODEL_NAME)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 512)
        
        self.image_encoder = load_resnet50()
        self.image_proj = nn.Linear(2048, 512)  # ResNet50输出特征维度为2048
        
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, image):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_output.pooler_output)

        image_feat = self.image_proj(self.image_encoder(image))
        
        fused_feat = torch.cat([text_feat, image_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        return self.classifier(fused_feat)

class EarlyFusionModel(nn.Module):
    def __init__(self, dropout_rate=BASE_DROPOUT_RATE):
        super(EarlyFusionModel, self).__init__()
        self.text_encoder = BertModel.from_pretrained(TEXT_MODEL_NAME)
        # 文本投影层的输出维度应为 1*224*224
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 224*224)
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 融合后的通道数是64+1=65，特征图尺寸是112x112
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(65, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        
        # 融合卷积后的输出维度是128*56*56=401408
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, image):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_output.last_hidden_state  # [B, seq_len, hidden_size]
        text_feat = torch.mean(text_embeds, dim=1)  # [B, hidden_size]（BERT-base的hidden_size是768）
        
        text_feat = self.text_proj(text_feat).reshape(-1, 1, 224, 224)  # size是768→投影到224*224=50176
        
        image_feat = self.image_conv(image)  # [B,64,112,112]
        text_feat = nn.Upsample(scale_factor=0.5, mode='bilinear')(text_feat)  # [B,1,112,112]
        
        fused_feat = torch.cat([image_feat, text_feat], dim=1)  # [B,65,112,112]
        fused_feat = self.fusion_conv(fused_feat)
        
        return self.classifier(fused_feat)

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, dropout_rate=BASE_DROPOUT_RATE):
        super(CrossAttentionFusionModel, self).__init__()
        # 文本分支
        self.text_encoder = BertModel.from_pretrained(TEXT_MODEL_NAME)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 512)
        
        # 图像分支
        self.image_encoder = load_resnet50()
        self.image_proj = nn.Linear(2048, 512)
        
        # 交叉注意力层（文本引导图像特征加权）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, image):
        # 文本特征提取
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_output.pooler_output).unsqueeze(1)  # [B, 1, 512]
        
        # 图像特征提取
        image_feat = self.image_proj(self.image_encoder(image)).unsqueeze(1)  # [B, 1, 512]
        
        # 交叉注意力融合（文本作为Query，图像作为Key/Value）
        attn_output, _ = self.cross_attention(
            query=text_feat,
            key=image_feat,
            value=image_feat,
            need_weights=False
        )
        fused_feat = attn_output.squeeze(1)  # [B, 512]
        
        return self.classifier(fused_feat)