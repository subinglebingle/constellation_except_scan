from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import models
import datasets
import config

import matplotlib.pyplot as plt
import wandb
wandb.init(project="Constellation")
#-----------------------------------------------------------------------------------------------------------------------------------
input_dim = 262144 #차원 맞추기(16*128*128) 
hidden_dim = 128
output_dim = 64
latent_dim = 16
r_dim = 16 #input_dim을 r_dim으로 차원 축소(인코더를 통해)
height = 128  # Height of the input image
width = 128  # Width of the input image

#상황에 따라 유동적인 변경 가능
conf = config.sprite_config

batch_size = conf.batch_size
num_epochs = conf.num_epochs

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
if use_cuda:
    print('------CUDA USED------')

loss_fn = models.LossFunctions(conf, beta=1, gamma_init = 0.1, latent_dim=16).to(device)   # 손실 함수 설정
os.makedirs(os.path.dirname(conf.checkpoint_dir), exist_ok=True) #..................checkpoint file 경로 생성

num_classes=379

# DataParallel에서 저장된 모델 체크포인트 로드 시 'module.' 접두사 제거 함수
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # 'module.' 접두사 제거
        new_state_dict[new_key] = value
    return new_state_dict


# Transform 적용 
# transforms.Compose(): 여러개의 전처리(transform)을 순차적으로 묶어서 한번에 적용해줌
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())
])

#데이터셋 있으면 있는거 사용, 없으면 make_sprites
dataset = datasets.Sprites(conf.data_dir, n=100000, canvas_size=128, train=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

monet = models.Monet(conf, height, width).to(device)
monet = nn.DataParallel(monet)

# monet pt파일 로드
if os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored Monet parameters from', conf.checkpoint_file)

else:
    print('Could not restore Monet parameters from', conf.checkpoint_file)

model = models.Constellation(conf, monet, latent_dim, hidden_dim, output_dim, latent_dim, r_dim, height, width).to(device) #input 차원 맞추기

optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

# monet 학습 생략 (이미 따로 학습된 ckpt파일 있음)
#train_monet(monet, data_loader, optimizer_monet)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정

# constellation pt 로드
pt_path = os.path.join(conf.checkpoint_dir, 'constellation.ckpt')

#model.load_state_dict(remove_module_prefix(checkpoint['constellation_model_state_dict']))
if os.path.isfile(pt_path):
    model.load_state_dict(torch.load(pt_path))
    print(f'Restored Constellation parameters from {pt_path}')
else: #없으면 train constellation
    print('Start training Constellation...')
    for epoch in range(num_epochs): #constellation 루프
        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda()

            # Constellation encode
            r, mu_q, logvar_q, a, o, learned_mask, recon, residue = model.encode(x)

            # Decode
            full_recon, mu, logvar = model.decode(r)

            optimizer.zero_grad()
            loss = loss_fn.total_loss(a, full_recon, mu_q, logvar_q, o, learned_mask)
            loss = torch.sum(loss)
            loss.backward()

            optimizer.step()

            #writer.add_scalar('Loss/train_constellation', loss.item(), epoch)  # 텐서보드에 기록
            wandb.log({"Loss/train_constellation": loss.item(), "epoch": epoch}) #wandb기록

        print(f'Epoch {epoch+1}/{num_epochs}, Constellation Loss: {loss.item()}')
        torch.save({'constellation_model_state_dict': model.state_dict()},'./checkpoints/constellation.ckpt')

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in model.parameters():
    param.requires_grad = False  # Constellation 모델 고정