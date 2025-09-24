from torch.utils.data import Dataset
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import models_vae as models
import datasets
import config
conf=config.sprite_config

import matplotlib.pyplot as plt
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, default=conf.beta)
args = parser.parse_args()

# sprite_config를 새로 만들어서 beta만 덮어쓰기
conf = conf._replace(beta=args.beta)

print("Training with beta =", conf.beta)

#config의 num_data랑 beta값 바꾸기,,,,

#--------------------------------------------------------------------------------------------------
latent_dim = conf.latent_dim #16
r_dim = conf.r_dim #16
height = 128  # Height of the input image
width = 128  # Width of the input image

batch_size = conf.batch_size
num_epochs = conf.num_epochs

device='cpu'

loss_fn = models.LossFunctions(conf).to(device)   # 손실 함수

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())
])

#데이터셋 있으면 있는거 사용, 없으면 make_sprites
dataset = datasets.Sprites(conf.data_dir, mode='test', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True)

monet = models.Monet(conf, height, width).to(device)
constellation = models.Constellation(conf, monet).to(device)

optimizer_monet = optim.RMSprop(monet.parameters(), lr=1e-5) #monet 논문: 1e-4, taming vae논문: 1e-5
optimizer = optim.Adam(list(constellation.parameters()) + list(loss_fn.parameters()), lr=1e-4) #논문:1e-3에서 1e-4로 스케줄링,, taming vae논문


# ckpt 불러오기 함수 (DataParallel 대응)
from collections import OrderedDict

def load_ckpt(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # DataParallel로 학습했으면 "module." prefix 제거
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"Restored parameters from {ckpt_path}")


# Monet ckpt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, f"monet_{conf.data_num}_{conf.beta}.ckpt")
if os.path.isfile(ckpt_path):
    load_ckpt(monet, ckpt_path)
else:
    print("No file in", ckpt_path)

# Constellation ckpt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, f"cons_{conf.data_num}_{conf.beta}.ckpt")
if os.path.isfile(ckpt_path):
    load_ckpt(constellation, ckpt_path)
else:
    print("No file in", ckpt_path)


def vis_final(images, save_dir='./vis_final', prefix='fianl'):
    os.makedirs(save_dir, exist_ok=True)
    num_slots=conf.num_slots

    monet.eval()
    constellation.eval()

    with torch.no_grad():  
        r, mu_q, logvar_q, a, o, learned_mask, masks_list = constellation.encode(images)
        ahat, mu, logvar = constellation.decode(r)
        #loss,rec_loss = loss_fn.total_loss(a, ahat, mu_q, logvar_q, o, learned_mask)

    for i in range(3):
        #print(ahat[i])
        fig, axs = plt.subplots(4, num_slots, figsize=(8,num_slots))

        # 1행: 원본, 전체 마스크 합, 재구성
        axs[0, 0].imshow(images[i].permute(1, 2, 0).detach())
        axs[0, 0].set_title('Input')
        axs[0, 0].axis('off')

        a_preds = []
        a_full_reconstruction = torch.zeros_like(images)
        for t, mask in enumerate(masks_list):
            sigma = conf.bg_sigma if t == 0 else conf.fg_sigma
            _, a_recon, a_pred = monet.decoder_step(x, o[:,t]*a[:,t], mask, sigma)
            a_preds.append(a_pred)
            a_full_reconstruction += mask * torch.clamp(a_recon, -10, 10)

        axs[0, 1].imshow((a_full_reconstruction[i]).detach().numpy().transpose(1, 2, 0))
        axs[0, 1].set_title('a')
        axs[0, 1].axis('off')
        
        ahat_preds = []
        ahat_full_reconstruction = torch.zeros_like(images)
        for t, mask in enumerate(masks_list):
            sigma = conf.bg_sigma if t == 0 else conf.fg_sigma
            _, ahat_recon, ahat_pred = monet.decoder_step(x, o[:,t]*ahat[:,t], mask, sigma)
            ahat_preds.append(ahat_pred)
            ahat_full_reconstruction += mask * torch.clamp(ahat_recon, -10, 10)

        axs[0, 2].imshow((ahat_full_reconstruction[i]).detach().numpy().transpose(1, 2, 0))
        axs[0, 2].set_title('a hat')
        axs[0, 2].axis('off')

        new_ahat_preds=[]
        new_ahat_full_reconstruction=torch.zeros_like(images)
        new_ahat = ahat.clone() # 새로운 tensor 생성 (원본 유지)
        new_ahat[..., 3] = new_ahat[..., 3] - 5  # 마지막 차원 첫 번째 요소만 -1
        print("ahat")
        print(ahat[i])
        print()
        print("new a hat")
        print(new_ahat[i])
        print()

        for t, mask in enumerate(masks_list):
            sigma = conf.bg_sigma if t == 0 else conf.fg_sigma
            _, new_ahat_recon, new_ahat_pred = monet.decoder_step(x, o[:,t]*new_ahat[:,t], mask, sigma)
            new_ahat_preds.append(new_ahat_pred)
            new_ahat_full_reconstruction += mask * torch.clamp(new_ahat_recon, -10, 10)

        axs[0, 3].imshow((new_ahat_full_reconstruction[i]).detach().numpy().transpose(1, 2, 0))
        axs[0, 3].set_title('new a hat')
        axs[0, 3].axis('off')

        # 1행의 [4:] 칸은 흰색 화면으로
        for col in range(4, 8):
            axs[0, col].axis('off')
            axs[0, col].set_facecolor("white")

        # 2행: masks_list
        for slot_idx in range(num_slots): 
            slot_mask = masks_list[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            axs[1, slot_idx].imshow(slot_mask[i].squeeze().detach().numpy(), cmap="gray") # , cmap="gray")
            axs[1, slot_idx].set_title(f'Mask {slot_idx}')
            axs[1, slot_idx].axis('off')

        # 3행: oa
        for slot_idx in range(num_slots): 
            slot_mask = a_preds[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            axs[2, slot_idx].imshow(slot_mask[i].detach().numpy(), cmap="gray")
            axs[2, slot_idx].set_title(f'a {slot_idx}')
            axs[2, slot_idx].axis('off')

        # 4행: oahat
        for slot_idx in range(num_slots): 
            slot_mask = ahat_preds[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            axs[3, slot_idx].imshow(slot_mask[i].detach().numpy(), cmap="gray")
            axs[3, slot_idx].set_title(f'a hat {slot_idx}')
            axs[3, slot_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{conf.beta}_{prefix}_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved FINAL visualization to {save_path}")


for batch in data_loader:
    x,_=batch

    vis_final(x)

    print()
    print('Finished...')
    
    break
    