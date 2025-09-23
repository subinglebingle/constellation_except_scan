# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch.distributions as dists
import torchvision
from torchvision import models, transforms
from PIL import Image
import config
conf=config.sprite_config

from torch_geometric.nn import GCNConv
from scipy.optimize import linear_sum_assignment
import itertools

#from monet import Monet as monet

import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MONet
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
    #convolution + batch_normalization + ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size,stride=stride,padding=padding,
                            bias=True)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbr=nn.Sequential(*layers)
            return cbr
        
        #enc
        self.enc1_1=CBR2d(in_channels=4,out_channels=64) #원래 in_channels=1 #4=3channels+1scope
        self.enc1_2=CBR2d(in_channels=64,out_channels=64)

        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.enc2_1=CBR2d(in_channels=64, out_channels=128)
        self.enc2_2=CBR2d(in_channels=128,out_channels=128)

        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.enc3_1=CBR2d(in_channels=128, out_channels=256)
        self.enc3_2=CBR2d(in_channels=256,out_channels=256)

        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.enc4_1=CBR2d(in_channels=256, out_channels=512)
        self.enc4_2=CBR2d(in_channels=512,out_channels=512)

        self.pool4=nn.MaxPool2d(kernel_size=2)

        self.enc5_1=CBR2d(in_channels=512,out_channels=1024)

        #dec
        self.dec5_1=CBR2d(in_channels=1024,out_channels=512)

        self.unpool4=nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2=CBR2d(in_channels=2*512,out_channels=512) #in_channels가 두배인 이유는 encoder의 일부가 붙기때문(skip connection)
        self.dec4_1=CBR2d(in_channels=512,out_channels=256)
    
        self.unpool3=nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2=CBR2d(in_channels=2*256,out_channels=256) 
        self.dec3_1=CBR2d(in_channels=256,out_channels=128)

        self.unpool2=nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2=CBR2d(in_channels=2*128,out_channels=128) 
        self.dec2_1=CBR2d(in_channels=128,out_channels=64)

        self.unpool1=nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2=CBR2d(in_channels=2*64,out_channels=64) 
        self.dec1_1=CBR2d(in_channels=64,out_channels=64)
        
        self.fc=nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1,stride=1,padding=0,bias=True) #원래 out_channels=1

    def forward(self,x):
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)

        enc2_1=self.enc2_1(pool1)
        enc2_2=self.enc2_2(enc2_1)
        pool2=self.pool2(enc2_2)

        enc3_1=self.enc3_1(pool2)
        enc3_2=self.enc3_2(enc3_1)
        pool3=self.pool3(enc3_2)

        enc4_1=self.enc4_1(pool3)
        enc4_2=self.enc4_2(enc4_1)
        pool4=self.pool4(enc4_2)

        enc5_1=self.enc5_1(pool4)

        dec5_1=self.dec5_1(enc5_1)

        unpool4=self.unpool4(dec5_1)
        cat4=torch.cat([unpool4, enc4_2], dim=1) #dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2=self.dec4_2(cat4)
        dec4_1=self.dec4_1(dec4_2)

        unpool3=self.unpool3(dec4_1)
        cat3=torch.cat([unpool3, enc3_2],dim=1)
        dec3_2=self.dec3_2(cat3)
        dec3_1=self.dec3_1(dec3_2)

        unpool2=self.unpool2(dec3_1)
        cat2=torch.cat([unpool2,enc2_2], dim=1)
        dec2_2=self.dec2_2(cat2)
        dec2_1=self.dec2_1(dec2_2)

        unpool1=self.unpool1(dec2_1)
        cat1=torch.cat([unpool1, enc1_2], dim=1)
        dec1_2=self.dec1_2(cat1)
        dec1_1=self.dec1_1(dec1_2)

        x=self.fc(dec1_1)
        return x #(16,2,128,128)

class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet().to(device)

    def forward(self, x, scope): #x: (16,3,128,128) 
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp) #logits: 16,2,128,128
        #alpha = torch.softmax(logits, 1) #16,2,128,128
        log_alpha = F.log_softmax(logits, dim=1)   #16,2,128,128   # 로그 확률(log softmax) 계산

        # # output channel 0 represents alpha_k,
        # # channel 1 represents (1 - alpha_k).
        # mask = scope + alpha[:, 0:1] #16,1,128,128
        # new_scope = scope + alpha[:, 1:2] #16,1,128,128

        # return mask, new_scope

        # log domain 변환
        eps = 1e-6
        log_scope = (scope+eps).log()              # scope -> log scope
        log_mask = log_scope + log_alpha[:, 0:1]    # log m_k
        new_log_scope = log_scope + log_alpha[:, 1:2]  # log s_k

        # 다시 확률 domain으로 변환해서 반환
        mask = log_mask.exp()
        new_scope = new_log_scope.exp()

        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4): 
            width = (width - 1) // 2
            height = (height - 1) // 2

        fc_in = 64 * width * height

        self.mlp = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True)
        )

        latent_dim = conf.latent_dim
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x=self.mlp(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class DecoderNet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(conf.latent_dim+2, 32, 3), # +2인 이유: y/x좌표 2채널(공간 정보)(coord_map)을 더해줬기 때문 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8) #padding 없이 convolution하기 위해
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs, indexing='xy')
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result
    
def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
# class VAE(nn.Module):
#     def __init__(self, width, height):
#         super().__init__()
#         self.encoder = EncoderNet(width, height)
#         self.decoder = DecoderNet(height, width)

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = reparameterize(mu, logvar)
#         out = self.decoder(z)
#         return out, mu, logvar

class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.beta = conf.beta
        self.gamma = conf.gamma

    def forward(self, x): #x: 64/4, 3, 128, 128 (16,3,128,128)
        scope = torch.ones_like(x[:, 0:1])
        # scope = torch.zeros_like(x[:, 0:1])  # log(1) = 0, 로그 공간 초기화
        masks = []
        zs=[] #...............latent vector 모음 (constellation을 위해 추가)
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)

        masks.append(scope) #num_slot-1+1 #8,16,1,128,128

        loss = torch.zeros_like(x[:, 0, 0, 0]) #16,
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)

        eps = 1e-6  # log 안정화용

        for i, mask in enumerate(masks):
            z, kl_z = self.encoder_step(x, mask)
            zs.append(z) #............latent vector 모음 (constellation을 위해 추가)
           
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred) #mask_pred: batch,128,128   #mask_preds:8,batch,128,128
            
            # --- 안정적 reconstruction loss ---
            # p_x: (B,H,W,C) 픽셀별 log_prob
            recon_loss = -p_x
            for d in range(1, p_x.dim()):  # batch 제외한 나머지 dim 평균
                recon_loss = recon_loss.mean(dim=d, keepdim=False)
            loss += recon_loss + self.beta * kl_z
            p_xs += recon_loss
            kl_zs += kl_z
            # full reconstruction
            full_reconstruction += mask * torch.clamp(x_recon, -10, 10)
           
            # loss += -p_x + self.beta * kl_z
            # p_xs += -p_x
            # kl_zs += kl_z
            # full_reconstruction += mask * x_recon
            
        # print("p_xs:", torch.mean(p_xs))
        # print()
        # print("beta * kl_zs",torch.mean(self.beta*kl_zs))
        # print()


        #zs: 8,batch,16
        zs=torch.stack(zs,1) #...........latent vector 모음 차원 합치기
        #zs: batch,8,16
    
        # masks 리스트를 그대로 tensor로 concat하기 전 상태로 저장
        masks_list = masks.copy() #deepcopy? 흠.,

        masks = torch.cat(masks, 1) #masks: batch,8,128,128

        tr_masks = masks.permute(0, 2, 3, 1) #tr_masks: batch,128,128,8
        tr_masks = tr_masks.clamp(min=1e-8, max=1.0)

        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3)) #logits: batch,128,128,8

        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])

        loss += self.gamma * kl_masks

        # print("gamma*kl_masks:", torch.mean(self.gamma*kl_masks))
        # print()
        # print("loss:", torch.mean(loss))
        # print()

        return {'loss':loss,
                'masks': masks,           # 합쳐진 마스크 batch,8,128,128
                'masks_list': masks_list, # 합치기 전 리스트 8,batch,1,128,128 #1은 channel (color channel=3)
                'reconstructions': full_reconstruction,
                'zs': zs #........latent vector 모음 #batch, 8, latent_dim       (64,8,16)
                }

    def encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        mu, logvar = self.encoder(encoder_input)  # (batch, latent_dim)
        z = reparameterize(mu, logvar)
        kl_z = compute_kl(mu, logvar)

        return z, kl_z
    #z: batch, latent_dim                 (64,16)
    #kl_z: batch


    def decoder_step(self, x, z, mask, sigma): #z: (batch, latent)
        decoder_output = self.decoder(z) #batch,4,128,128
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]

        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)

        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=p_x.dtype, device=p_x.device)

        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])

        return p_x, x_recon, mask_pred
        # p_x: batch
        # x_recon: batch,3,128,128
        # mask_pred: batch,128,128

class MaskExtractor(nn.Module):
    def __init__(self, num_slots=conf.num_slots, latent_dim=conf.latent_dim):
        super().__init__()
        self.num_slots = num_slots
        self.latent_dim = latent_dim
        # mask extractor: input (B, num_slots, latent_dim) → output (1, num_slots, latent_dim) or (1, num_slot, 1)
        self.mask_extractor = nn.Sequential(
            #nn.Conv1d(num_slots, 16, 3, padding=1),
            nn.Conv1d(latent_dim, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 1), # nn.Conv1d(32, 16, 1)는 soft assignment, nn.Conv1d(32, 1, 1) 은 hard assignment
            nn.Softmax(dim=2)   # latent_dim 축에 대해 softmax
        )

    def forward(self, entities):
        # entities: (batch, num_slots, latent_dim)
        batch, num_slots, latent_dim = entities.shape
        
        assert num_slots == self.num_slots and latent_dim == self.latent_dim

        # Conv1d input: (B, num_slots, latent_dim) → Conv1d expects (B, in_channels, length)
        x = entities  # (B, num_slots, latent_dim)
        x = x.transpose(1, 2) #(B, latent_dim, num_slots)
        # Conv1d 입력은 (B, in_channels, L) 구조(슬롯이 채널)
        masks = self.mask_extractor(x)  # (B, 1, latent_dim) #soft? hard? assignment
        masks=masks.transpose(1,2)
        # 마스크 (B, num_slots, 1) → (B, num_slots, latent_dim)로 브로드캐스팅 곱
        # print("masks:", masks.shape)
        # print("entities:", entities.shape)
        weighted_entities = entities * masks  # 자동 broadcast
        # print("weighted_entities:", weighted_entities.shape)
        return weighted_entities, masks  # 각각 (batch, num_slots, latent_dim), (batch, num_slots, latent_dim)

 #여기부터 GNN 코드를 논문에 나온대로
class EdgeMLP(nn.Module):
    def __init__(self,input_dim=2*conf.latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    def forward(self, edge_features):
        return self.mlp(edge_features) #(batch_size, num_slots*num_slots, 64)

class NodeMLP(nn.Module):
    def __init__(self,input_dim=conf.latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
    def forward(self, node_features):
        return self.mlp(node_features) #(batch_size, num_slots, 128)

class GlobalsMLP(nn.Module):
    def __init__(self, r_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128+64, 256), #torch.cat([node_summary, edge_summary], dim=-1)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, r_dim*2)  # µq, σq (각 r_dim 만큼)
        )
    def forward(self, globals_features):
        return self.mlp(globals_features)

class GNNInferenceNetwork(nn.Module):
    def __init__(self, r_dim=conf.r_dim):
        self.r_dim=r_dim
        super().__init__()

        self.edge_mlp = EdgeMLP()
        self.node_mlp = NodeMLP()

        self.edge_update_mlp= EdgeMLP(input_dim=320)
        self.node_update_mlp= NodeMLP(input_dim=192)

        self.globals_mlp = GlobalsMLP(r_dim)
        self.num_message_passing_steps = 2

    def forward(self, node_features):
        # 엣지설정    
        edge_features = torch.zeros(conf.batch_size, conf.num_slots, conf.num_slots, conf.latent_dim * 2, device=node_features.device)

        for i in range(conf.num_slots):
            for j in range(conf.num_slots):
                # node_features[:, i, :] : (batch_size, latent_dim)
                # node_features[:, j, :] : (batch_size, latent_dim)
                edge = torch.cat([node_features[:, i, :], node_features[:, j, :]], dim=-1)  # (batch_size, 2*latent_dim)
                edge_features[:, i, j, :] = edge  # (batch_size, 2*latent_dim) 위치에 할당

        # 결과: edge_features.shape == (batch_size, num_slots, num_slots, 2*latent_dim)=(128,8,8,32)

        #MLP 통과
        node_embeddings = self.node_mlp(node_features)   # (128, 8, 128)
        edge_embeddings = self.edge_mlp(edge_features)   # (128, 8, 8, 64)

        # 각 배치별 sum aggregation (mean, max 등 다른 pooling 가능)
        node_summary = node_embeddings.sum(dim=1)   # (128, 128)
        edge_summary = edge_embeddings.sum(dim=(1, 2))    # (128, 64)
        global_features = torch.cat([node_summary, edge_summary], dim=-1)  # (128, 192)

        for _ in range(self.num_message_passing_steps):
            node_embeddings, edge_embeddings, global_features = self.message_passing(node_embeddings, edge_embeddings, global_features)
        # globals_에서 µq, σq 추출
        mu, logvar = global_features[:, :self.r_dim], global_features[:, self.r_dim:]
        sigma = F.softplus(logvar) #로그분산(logvar) 값을 양수 표준편차(sigma)로 변환하기 위해 사용되고, 연속적이고 부드럽게 0 이상인 값을 반환해 안정적인 분산 추정이 가능
        return mu, sigma 
    
    def aggregate_neighbor_messages(self, edge_feat):
        '''
        edge_feat: (batch, num_nodes, num_nodes, edge_dim)
        - [b, i, j, d]: batch b에서 노드 i가 노드 j로부터 받은 edge 임베딩 (i←j)
        - 모든 노드쌍(fully-connected)라면, 각 행은 자신의 이웃(전체)로부터 정보 집계
        리턴: (batch, num_nodes, edge_dim)  # 각 노드당 메시지 합산
        '''
        # 보통 자기 자신 edge(i,i)는 제외하거나, 전부 합산해도 무방 (선택)
        messages = edge_feat.mean(dim=2)    # i번 노드가 j(모든 이웃)에서 받은 메시지 합산
        return messages

    def global_readout(self, feat):
        '''
        feat: (batch, num_items, feat_dim)
        - 노드(=num_items=num_nodes), 또는 엣지(=num_edges 등) 임베딩
        리턴: (batch, feat_dim)  # 그래프별 summary vector
        '''
        # 합 pooling (노드 수 고정이면 sum이 강한 표현!)
        return feat.mean(dim=1)
    
    def message_passing(self, node_embeddings, edge_embeddings, global_features):
        '''
        Inputs:
        node_embeddings: (batch_size, num_nodes, node_feat_dim)       # (128, 8, 128)
        edge_embeddings: (batch_size, num_nodes, num_nodes, edge_feat_dim)  # (128, 8, 8, 64)
        global_features: (batch_size, global_feat_dim)
        '''

        # 1. 노드 메시지 집계 - 이웃 엣지 임베딩 합산
        node_messages = self.aggregate_neighbor_messages(edge_embeddings)  # (128, 8, 64)

        # 2. 기존 노드 임베딩과 메시지를 concat하여 노드 업데이트
        combined_node_input = torch.cat([node_embeddings, node_messages], dim=-1)  # (128, 8, 128+64=192)
        updated_node_embeddings = self.node_update_mlp(combined_node_input)  # (128, 8, 128)

        # 3. 노드 임베딩을 활용하여 엣지 업데이트

        # 노드 임베딩 쌍 생성: (batch, N, N, node_feat_dim)
        node_i = updated_node_embeddings.unsqueeze(2).expand(-1, -1, updated_node_embeddings.size(1), -1)  # (128, 8, 8, 128)
        node_j = updated_node_embeddings.unsqueeze(1).expand(-1, updated_node_embeddings.size(1), -1, -1)  # (128, 8, 8, 128)

        # 엣지 MLP 입력 생성: 기존 edge + node_i + node_j concat
        edge_inputs = torch.cat([edge_embeddings, node_i, node_j], dim=-1)  # (128, 8, 8, 64 + 128*2 = 320)
        updated_edge_embeddings = self.edge_update_mlp(edge_inputs)  # (128, 8, 8, 64)

        # 4. 글로벌 피처 요약 (평균 pooling 권장)
        node_summary = self.global_readout(updated_node_embeddings)  # (128, 128)
        edge_summary = self.global_readout(self.aggregate_neighbor_messages(updated_edge_embeddings))  # (128, 64)

        global_inputs = torch.cat([node_summary, edge_summary], dim=-1)  # (128, 192)

        # self.globals_mlp는 input_dim=192, output_dim=global_features.shape[-1] 사전설계 필요
        updated_global_features = self.globals_mlp(global_inputs)  # (128, r_dim*2)

        return updated_node_embeddings, updated_edge_embeddings, updated_global_features

# LSTM
class SequentialLSTM(nn.Module):
    def __init__(self, r_dim=conf.r_dim, hidden_dim=conf.hidden_dim, latent_dim=conf.latent_dim, num_slots=conf.num_slots, batch_size=conf.batch_size):
        super(SequentialLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(r_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mu 추출
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # logvar 추출
        self.num_slots = num_slots
        self.batch_size= batch_size

    def forward(self, r): #r: representation
        h_t, c_t = self.init_hidden(self.batch_size, r.device)

        mu_outputs = []
        logvar_outputs = []
        for i in range(self.num_slots):
            h_t, c_t = self.lstm_cell(r, (h_t, c_t))  # 각 슬롯에 대해 LSTM 처리
            mu_i = self.fc_mu(h_t)  # mu 추출
            logvar_i = self.fc_logvar(h_t)  # logvar 추출
            mu_outputs.append(mu_i)
            logvar_outputs.append(logvar_i)

        # [batch_size, num_slots, latent_dim] 형태로 출력 쌓기
        mu_outputs = torch.stack(mu_outputs, dim=1)
        logvar_outputs = torch.stack(logvar_outputs, dim=1)
        return mu_outputs, logvar_outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_hidden(self, batch_size,device):
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        return h, c


#Constellation
class Constellation(nn.Module):
    def __init__(self, conf, monet):
        super(Constellation, self).__init__()
        self.monet = monet
        self.mask_extractor=MaskExtractor()
        self.node_mlp = NodeMLP()
        self.edge_mlp = EdgeMLP()
        self.gnn = GNNInferenceNetwork() 
        self.lstm = SequentialLSTM(conf.r_dim, conf.hidden_dim, conf.latent_dim, conf.num_slots)
    
    def encode(self, x):
        monet_output = self.monet(x)
        masks_list=monet_output['masks_list'] #(8,64,1,128,128)
        o = monet_output['zs'] #(128,8,16)

    #mask extractor & a는 learned mask 통과한 zs
        a, learned_mask = self.mask_extractor(o)

    #gnn
        mu_q, logvar_q = self.gnn(a) #node_features=a # (128, 8, 16)

    #lstm 
        r = self.lstm.reparameterize(mu_q, logvar_q)
        return r, mu_q, logvar_q, a, o, learned_mask, masks_list

    def decode(self, r):
        mu_outputs, logvar_outputs = self.lstm(r)
        a_hat = self.lstm.reparameterize(mu_outputs, logvar_outputs)
        return a_hat, mu_outputs, logvar_outputs

class LossFunctions(nn.Module):
    def __init__(self, conf, beta=4, gamma_init=0.1, latent_dim=conf.latent_dim):
        super(LossFunctions, self).__init__()
        self.beta = beta
        self.gamma = nn.Parameter(torch.tensor([gamma_init] * conf.num_slots * latent_dim))

    # 논문에 나온 손실 함수 구현
    def reconstruction_loss(self, ai, a_hat):
        loss = 0.0
        batch_size=conf.batch_size
        num_slots= conf.num_slots

        for b in range(batch_size):
            # 각 배치에 대해 cost_matrix를 생성
            cost_matrix = np.zeros((num_slots, num_slots))
            for i in range(num_slots):
                for j in range(num_slots):
                    cost_matrix[i, j] = np.linalg.norm(ai[b, i].detach().cpu().numpy() - a_hat[b, j].detach().cpu().numpy())

            # Hungarian matching (linear_sum_assignment) 적용
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 매칭된 슬롯 쌍에 대해 재구성 손실 계산
            for i, j in zip(row_ind, col_ind):
                loss += 0.5 * torch.norm(ai[b, i] - a_hat[b, j]) ** 2

        return loss

    def kl_divergence(self, mu_q, logvar_q, mu_p=None, logvar_p=None):
        if mu_p is None or logvar_p is None:
            mu_p = torch.zeros_like(mu_q)
            logvar_p = torch.zeros_like(logvar_q)
        kld = -0.5 * torch.sum(1 + logvar_q - logvar_p - ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p))
        return kld

    def mask_entropy_loss(self, learned_mask):
        # 배치와 슬롯 차원을 모두 포함하여 손실 계산
        loss = -torch.sum(learned_mask * torch.log(learned_mask + 1e-10)) 
        return -loss # ..........교수님께서 논문에 부호오류라고 하신 곳

    def conditioning_loss(self, o, a_hat, learned_mask):
        loss = 0.0
        batch_size, num_slots, latent_dim = o.size()

        # learned_mask의 차원을 num_slots에 맞춰서 반복 (batch_size, num_slots, latent_dim)
        learned_mask = learned_mask.repeat(1, num_slots, 1)

        for b in range(batch_size):
            cost_matrix = np.zeros((num_slots, num_slots))

            # 각 배치 내에서 o와 a_hat 간의 cost_matrix 계산
            for i in range(num_slots):
                for j in range(num_slots):
                    cost_matrix[i, j] = np.linalg.norm(
                        o[b, i].detach().cpu().numpy() - a_hat[b, j].detach().cpu().numpy() / learned_mask[b, j].detach().cpu().numpy()
                    )

            # Hungarian matching (linear_sum_assignment) 적용
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 매칭된 쌍에 대해 손실 계산
            for i, j in zip(row_ind, col_ind):
                l_rec_star = 0.5 * torch.sum((o[b, i] - a_hat[b, j] / learned_mask[b, j]) ** 2)
                gamma_j = self.gamma[j]  # 학습 가능한 gamma 변수
                loss += torch.sum((1 - learned_mask[b, j]) * torch.abs(l_rec_star - gamma_j))

        return loss

    def reordering_loss(self, a_hat):
        loss = 0.0
        batch_size, num_slots, latent_dim = a_hat.size()

        # 각 배치 내에서 인접한 슬롯들의 차이를 계산
        for b in range(batch_size):
            for i in range(1, num_slots):
                loss += torch.norm(a_hat[b, i] - a_hat[b, i - 1]) ** 2

        return loss

    def total_loss(self, ai, a_hat, mu_q, logvar_q, o, learned_mask):
        L_rec = self.reconstruction_loss(ai, a_hat)
        L_kl = self.beta * self.kl_divergence(mu_q, logvar_q)
        L_entropy = self.mask_entropy_loss(learned_mask)
        L_condition = self.conditioning_loss(o, a_hat, learned_mask)
        L_reorder = self.reordering_loss(a_hat)

        L_total = L_rec + L_reorder + L_kl + L_entropy + L_condition
        return L_total, L_rec
