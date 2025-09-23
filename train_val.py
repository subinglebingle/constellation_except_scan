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

wandb.init(project="Constellation without scan", name=f"{conf.data_num}, beta={conf.beta}")

print("Training with beta =", conf.beta)


#-----------------------------------------------------------------------------------------------------------------------------------
# input_dim = 262144 #차원 맞추기(16*128*128) 
# hidden_dim = conf.hidden_dim #128
# output_dim = 64
latent_dim = conf.latent_dim #16
r_dim = conf.r_dim #16
height = 128  # Height of the input image
width = 128  # Width of the input image

batch_size = conf.batch_size
num_epochs = conf.num_epochs

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
if use_cuda:
    print('------CUDA USED------')

loss_fn = models.LossFunctions(conf).to(device)   # 손실 함수 설정
os.makedirs(os.path.dirname(conf.checkpoint_dir), exist_ok=True) #..................checkpoint file 경로 생성

# transforms.Compose(): 여러개의 전처리(transform)을 순차적으로 묶어서 한번에 적용해줌
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())
])

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def train_monet(monet, data, optimizer, epoch):
    monet.train()

    if use_cuda:
        data = data.cuda()

    optimizer.zero_grad()
    output = monet(data)

    # masks_list=output["masks_list"] #8,64,1,128,128

    loss = torch.mean(output['loss'])
    loss.backward()
    optimizer.step()

    wandb.log({"Loss/train_monet": loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item()

def val_monet(monet, val_data, epoch):
    monet.eval()

    if use_cuda:
        val_data = val_data.cuda()

    with torch.no_grad():
        output = monet(val_data)
        loss = torch.mean(output['loss'])
        #loss = output['kl_loss'] + torch.exp(log_lambda) * c_t + output['masks_loss']
    
    loss=loss.mean()

    wandb.log({"Loss/val_monet": loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item(), output

def vis_monet(images, masks, reconstructions, masks_list=None, save_dir='./vis_monet', prefix='monet'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(3):
        # masks_list가 있으면 행 2, 없으면 행 1짜리
        num_mask_rows = 2 if masks_list is not None else 1
        fig, axs = plt.subplots(num_mask_rows, 8, figsize=(8,2))

        # 1행: 원본, 전체 마스크 합, 재구성
        axs[0, 0].imshow(images[i].transpose(1, 2, 0))
        axs[0, 0].set_title('Input')
        axs[0, 0].axis('off')

        # axs[0, 1].imshow(masks[i].sum(axis=0))
        # axs[0, 1].set_title('Mask(Sum)')
        # axs[0, 1].axis('off')

        axs[0, 1].imshow(reconstructions[i].transpose(1, 2, 0))
        axs[0, 1].set_title('Reconstruction')
        axs[0, 1].axis('off')

        # 1행의 [3:] 칸은 흰색 화면으로
        for col in range(2, 8):
            axs[0, col].axis('off')
            axs[0, col].set_facecolor("white")

        # 2행: masks_list가 있을 때만
        if masks_list is not None:
            # masks_list는 [num_slots][batch, C, H, W] 형태라 가정
            num_slots = len(masks_list)
            for slot_idx in range(num_slots): 
                slot_mask = masks_list[slot_idx][i].sum(axis=0)  # 채널 축 합쳐서 (H,W)로
                axs[1, slot_idx].imshow(slot_mask)
                axs[1, slot_idx].set_title(f'Mask {slot_idx}')
                axs[1, slot_idx].axis('off')
            # 만약 slots가 3보다 적으면 빈 칸은 없애기
            for empty_idx in range(num_slots, 3):
                axs[1, empty_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{conf.beta}_{prefix}_{epoch}th epoch_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved monet visualization to {save_path}")

def constraint_fn(recon_loss):  # L2 리컨 오차가 특정 값 이하로 유지
    return recon_loss.mean() - 0.5 #target_constraint: κ ∈ {0.06, 0.08, 0.1, 0.125, 0.175}

def train_constellation(constellation, data, optimizer, epoch, log_lambda, C_ma, constraint_fn, alpha=0.99, lr=1e-4):
    """
    GECO 알고리즘 한 배치 업데이트 함수

    Args:
        model: PyTorch 모델
        x: 입력 배치 (Tensor)
        optimizer: 모델 파라미터용 옵티마이저
        log_lambda: 라그랑주 승수의 로그값 (Tensor, requires_grad=True 아님)
        C_ma: 이전까지의 제약 조건 moving average (float 또는 Tensor)
        constraint_fn: 제약 조건을 계산하는 함수 (x, recon_x) -> Tensor
        alpha: moving average 하이퍼파라미터
        lr: 학습률 (옵티마이저 lr과 별도)
        device: 연산 디바이스

    Returns:
        loss_value: 현재 배치 손실(float)
        updated_log_lambda: 업데이트된 log_lambda (Tensor)
        updated_C_ma: 업데이트된 moving average (Tensor)
    """
    constellation.train()

    if use_cuda:
        data = data.cuda()

    optimizer.zero_grad()
    r, mu_q, logvar_q, a, o, learned_mask, masks_list = constellation.module.encode(data)
    # o=zs (128,8,16)
    
    a_hat, mu_outputs, logvar_outputs = constellation.module.decode(r)

    # print("a:", a.shape) # (64,8,16)
    # print("a_hat:",a_hat.shape) # (64,8,16)

    loss ,recon_loss= loss_fn.total_loss(a, a_hat, mu_outputs, logvar_outputs, o, learned_mask)

    C_hat=constraint_fn(recon_loss)
    if C_ma is None:
        C_ma_new=C_hat.detach()
    else: C_ma_new=alpha * C_ma + (1-alpha) * C_hat.detach()

    C_t = C_hat + (C_ma_new - C_hat).detach()
    
    loss=(loss-recon_loss)+ torch.exp(log_lambda)*C_t

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        log_lambda = log_lambda + lr * C_t.detach()

    wandb.log({"Loss/train_constellation": loss.item(), "epoch": epoch}) #wandb 기록
    wandb.log({"rec_Loss/train_constellation": recon_loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item(), recon_loss.item()

def val_constellation(constellation, val_data, epoch):
    constellation.eval()

    if use_cuda:
        val_data = val_data.cuda()

    with torch.no_grad():
        r, mu_q, logvar_q, a, o, learned_mask, masks_list = constellation.module.encode(val_data)
        a_hat, mu, logvar = constellation.module.decode(r)
        loss,rec_loss = loss_fn.total_loss(a, a_hat, mu_q, logvar_q, o, learned_mask)

    wandb.log({"Loss/val_constellation": loss.item(), "epoch": epoch}) #wandb 기록
    wandb.log({"rec_Loss/val_constellation": rec_loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item(), rec_loss.item(), a, r, a_hat, masks_list, o

def vis_constellation(images, a, r, a_hat, masks_list, o, save_dir='./vis_cons', prefix='cons'):
    os.makedirs(save_dir, exist_ok=True)
    num_slots=conf.num_slots
    for i in range(3):
        fig, axs = plt.subplots(4, num_slots, figsize=(8,num_slots))

        # 1행: 원본, 전체 마스크 합, 재구성
        #axs[0, 0].imshow(images[i].transpose(1, 2, 0))
        axs[0, 0].imshow(images[i].permute(1, 2, 0).cpu())
        axs[0, 0].set_title('Input')
        axs[0, 0].axis('off')

        # print("a;", a.shape) #batch,8,16
        # print("o;", o.shape) #batch,8,16
        #sigma = conf.bg_sigma if i == 0 else conf.fg_sigma
    
        # print("images shape:", images.shape) #128,3,128,128
        # masks_list shape= 8, 64, 1, 128, 128  #[num_slots,batch, C, H, W]

        # selected_masks = [masks_list[slot][i] for slot in range(len(masks_list))]
        # masks_i = torch.stack(selected_masks)  # shape (8, 1, 128, 128)

        a_preds = []
        a_full_reconstruction = torch.zeros_like(images)
        for t, mask in enumerate(masks_list):
            sigma = conf.bg_sigma if t == 0 else conf.fg_sigma
            #print("a[:,t] = ", a[:,t].shape) # 64,16
            _, a_recon, a_pred = monet.module.decoder_step(x, o[:,t]*a[:,t], mask, sigma)
            a_preds.append(a_pred)

            a_full_reconstruction += mask * torch.clamp(a_recon, -10, 10)

        #_,oa, _= monet.module.decoder_step(images[i], a[i], masks_i, sigma)
        #_,oa_recon, oa_pred = monet.module.decoder_step(images, o*a, masks_list, sigma)
        #print(oa.shape) #8, 3, 128, 128 

        # axs[0, 1].imshow((oa.sum(axis=0)).cpu().numpy().transpose(1, 2, 0))
        axs[0, 1].imshow((a_full_reconstruction[i]).cpu().numpy().transpose(1, 2, 0))
        axs[0, 1].set_title('a')
        axs[0, 1].axis('off')
        
        ahat_preds = []
        ahat_full_reconstruction = torch.zeros_like(images)
        for t, mask in enumerate(masks_list):
            sigma = conf.bg_sigma if t == 0 else conf.fg_sigma
            _, ahat_recon, ahat_pred = monet.module.decoder_step(x, o[:,t]*a_hat[:,t], mask, sigma)
            ahat_preds.append(ahat_pred)

            ahat_full_reconstruction += mask * torch.clamp(ahat_recon, -10, 10)
        print("ahat_preds:",len(ahat_preds),len(ahat_preds[0]),len(ahat_preds[0][0]),len(ahat_preds[0][0][0]))
        print("ahat_full_reconstruction",len(ahat_full_reconstruction),len(ahat_full_reconstruction[0]),len(ahat_full_reconstruction[0][0]))
        # _, oahat, _=monet.module.decoder_step(images[i], a_hat[i], masks_i, sigma)
        #_, oahat_recon, oahat_pred=monet.module.decoder_step(images, o*a_hat, masks_list, sigma)
        axs[0, 2].imshow((ahat_full_reconstruction[i]).cpu().numpy().transpose(1, 2, 0))
        axs[0, 2].set_title('a_hat')
        axs[0, 2].axis('off')

        # 1행의 [3:] 칸은 흰색 화면으로
        for col in range(3, 8):
            axs[0, col].axis('off')
            axs[0, col].set_facecolor("white")

        # 2행: masks_list
        for slot_idx in range(num_slots): 
            slot_mask = masks_list[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            #axs[1, slot_idx].imshow(slot_mask.cpu().numpy())
            axs[1, slot_idx].imshow(slot_mask[i].squeeze().cpu().numpy()) # , cmap="gray")
            axs[1, slot_idx].set_title(f'Mask {slot_idx}')
            axs[1, slot_idx].axis('off')

        # 3행: oa
        for slot_idx in range(num_slots): 
            slot_mask = a_preds[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            axs[2, slot_idx].imshow(slot_mask[i].cpu().numpy())
            axs[2, slot_idx].set_title(f'a {slot_idx}')
            axs[2, slot_idx].axis('off')

        # 4행: oahat
        for slot_idx in range(num_slots): 
            slot_mask = ahat_preds[slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            axs[3, slot_idx].imshow(slot_mask[i].cpu().numpy())
            axs[3, slot_idx].set_title(f'a_hat {slot_idx}')
            axs[3, slot_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{conf.beta}_{prefix}_{epoch}th epoch_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved constealltion visualization to {save_path}")

#데이터셋 있으면 있는거 사용, 없으면 make_sprites
dataset = datasets.Sprites(conf.data_dir, mode='train', transform=transform)
val_dataset= datasets.Sprites(conf.data_dir, mode='val', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

monet = nn.DataParallel(models.Monet(conf, height, width).to(device))
constellation = nn.DataParallel(models.Constellation(conf, monet).to(device)) #input 차원 맞추기 #dataparallel 추가함 오류나면 걍 빼
#cpu
# monet = (models.Monet(conf, height, width).to(device))
# constellation = (models.Constellation(conf, monet).to(device))


optimizer_monet = optim.RMSprop(monet.parameters(), lr=1e-5) #monet 논문: 1e-4, taming vae논문: 1e-5
optimizer = optim.Adam(list(constellation.parameters()) + list(loss_fn.parameters()), lr=1e-4) #논문:1e-3에서 1e-4로 스케줄링,, taming vae논문


# monet ckpt파일 로드
ckpt_path = os.path.join(conf.checkpoint_dir, f'monet_{conf.data_num}_{conf.beta}.ckpt')

if os.path.isfile(ckpt_path):
    monet.load_state_dict(torch.load(ckpt_path))
    print('Restored Monet parameters from', ckpt_path)
else:
    print('Start training Monet...')

    best_val_loss=float('inf')

    for epoch in range(num_epochs): #monet 루프
        train_loss=0
        # log_lambda = torch.tensor([0.0], device=device)  # 초기값 #77
        # C_ma = None #77
        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda() #x: 64,3,128,128
            
            # Train MONet
            #batch_loss, log_lambda, C_ma, c_t , _ = monet_geco_step(monet, x, optimizer_monet, log_lambda, C_ma, monet_constraint_fn, alpha=0.99, lr=1e-4) #77
            batch_loss=train_monet(monet, x, optimizer_monet, epoch)
            train_loss+=batch_loss

        train_loss/=len(data_loader)

        print(f'MONet Epoch {epoch+1}/{num_epochs}, loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader: 
            val_x, _ = val_batch
            if use_cuda:
                val_x = val_x.cuda()

            batch_loss, output=val_monet(monet, val_x, epoch)
            val_loss+=batch_loss

        val_loss/=len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(monet.state_dict(), ckpt_path)
            masks_list_npy = [numpify(slot[:8]) for slot in output['masks_list']]
            vis_monet(numpify(val_x[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]),
                                masks_list_npy)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정


# constellation pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, f'cons_{conf.data_num}_{conf.beta}.ckpt')

if os.path.isfile(ckpt_path):
    constellation.load_state_dict(torch.load(ckpt_path))
    print(f'Restored Constellation parameters from {ckpt_path}')
else: #없으면 train constellation
    print('Start training Constellation...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs): #train constellation 루프
        train_loss=0
        log_lambda = torch.tensor([0.0], device=device)  # 초기값 #77
        C_ma = None #77

        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda()

            # Train MONet
                
            batch_loss, recon_loss =train_constellation(constellation, x, optimizer, epoch, log_lambda, C_ma, constraint_fn)
            train_loss+=batch_loss

        train_loss/=len(data_loader)

        print(f"total loss: {batch_loss}")
        print(f"recon_loss: {recon_loss}")

        print(f'Constellation Epoch {epoch+1}/{num_epochs}, loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader: 
            val_x, _ = val_batch
            if use_cuda:
                val_x = val_x.cuda()

            batch_loss, rec_loss, a, r, a_hat, masks_list, o=val_constellation(constellation, val_x, epoch)
            val_loss+=batch_loss

        val_loss/=len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            #masks_list_npy = [numpify(slot[:8]) for slot in masks_list]
            vis_constellation(val_x, a, r, a_hat, masks_list, o)
            torch.save(constellation.state_dict(), ckpt_path)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in constellation.parameters():
    param.requires_grad = False  # Constellation 모델 고정

print('Train Fin.')