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

conf = config.sprite_config
#임시로 만든 하이퍼파라미터와 설정값들. 상황에 따라 유동적인 변경 가능

batch_size = conf.batch_size
num_epochs = conf.num_epochs

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
if use_cuda:
    print('------CUDA USED------')

loss_fn = models.LossFunctions(conf, beta=1, gamma_init = 0.1, latent_dim=16).to(device)   # 손실 함수 설정
os.makedirs(os.path.dirname(conf.checkpoint_dir), exist_ok=True) #..................checkpoint file 경로 생성

def change_factor_positions(ys1, ys2, factor_sizes):
    ys = torch.zeros_like(ys1)  # 출력할 멀티핫 인코딩 ys 생성
    start = 0

    # 각 팩터별로 ys1에서 1의 위치를 ys2의 1 위치로 변경
    for size in factor_sizes:
        end = start + size

        # ys1과 ys2에서 현재 팩터 구간 내 1의 위치를 찾기
        ys1_slice = ys1[:, start:end]
        ys2_slice = ys2[:, start:end]

        # torch.argmax()로 구간 내 1의 위치를 찾기
        ys1_pos = torch.argmax(ys1_slice, dim=1)  # 각 배치에 대해 ys1에서 1이 있는 위치
        ys2_pos = torch.argmax(ys2_slice, dim=1)  # 각 배치에 대해 ys2에서 1이 있는 위치

        # 변경이 있는지 확인하기 위한 플래그
        no_1_in_ys1 = (torch.sum(ys1_slice, dim=1) == 0)  # ys1 구간에 1이 없는 경우
        no_1_in_ys2 = (torch.sum(ys2_slice, dim=1) == 0)  # ys2 구간에 1이 없는 경우

        for i in range(ys.size(0)):  # 배치 크기만큼 반복
            max_index = start + size - 1  # 최대 인덱스는 현재 구간 내에서 끝 값으로 설정

            # 인덱스를 클램핑하여 최대 크기를 넘지 않도록 보장
            ys1_clamped = torch.clamp(start + ys1_pos[i], 0, max_index)
            ys2_clamped = torch.clamp(start + ys2_pos[i], 0, max_index)

            if no_1_in_ys2[i]:
                # ys2에 1이 없으면, 변경 없음 -> ys1의 값 유지
                if not no_1_in_ys1[i]:
                    # ys1에 1이 있으면, 그대로 ys1_clamped 사용
                    ys[i, ys1_clamped] = 1
            else:
                # ys2에 1이 있으면, 변경된 위치 반영 (클램핑 적용)
                if not no_1_in_ys1[i]:
                    ys[i, ys1_clamped] = 0  # 기존 ys1의 1을 0으로 설정
                ys[i, ys2_clamped] = 1  # ys2의 1 위치를 반영
        start = end

    return ys

def change_factor_randomly_with_zero_option(ys1, factor_sizes, zero_prob=0.5): #recombinator 학습용 개입 벡터 ys 만들기. 입력되는 multi hot encoding을 모델링
    # ys1과 동일한 크기의 영벡터 생성
    ys = torch.zeros_like(ys1)
    start = 0

    # 각 팩터별로 랜덤으로 1의 위치를 변경하거나 영벡터로 설정
    for size in factor_sizes:
        end = start + size

        # 일정 확률로 해당 팩터에 1을 배치할지 결정 (zero_prob 확률로 영벡터)
        if torch.rand(1).item() > zero_prob:  # 1 - zero_prob 확률로 1 배치
            random_pos = torch.randint(0, size, (1,)).item()
            ys[:,(start + random_pos)] = 1

        # 그렇지 않으면 영벡터로 유지

        start = end

    return ys

def train_monet(monet, data, optimizer):
    monet.train()

    if use_cuda:
        data = data.cuda()

    optimizer.zero_grad()
    output = monet(data)
    loss = torch.mean(output['loss'])
    loss.backward()
    optimizer.step()

    wandb.log({"Loss/train_monet": loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item()

save_dir = "./visualizations"
os.makedirs(save_dir, exist_ok=True)

def visualize_and_log(output_dict, epoch, prefix="monet_train"):
    recon = output_dict['reconstructions'].detach().cpu()
    masks = output_dict['masks'].detach().cpu()
    orig = output_dict.get('original_image', None)
    
    K = masks.shape[1]
    i = 0  # 첫 번째 샘플만

    images_to_log = []

    # 원본
    if orig is not None:
        orig_img = transforms.ToPILImage()(orig[i])
        orig_path = os.path.join(save_dir, f"{prefix}_epoch{epoch}_original.png")
        orig_img.save(orig_path)
        images_to_log.append(wandb.Image(orig_img, caption="Original"))

    # 재구성
    recon_img = transforms.ToPILImage()(recon[i])
    recon_path = os.path.join(save_dir, f"{prefix}_epoch{epoch}_reconstruction.png")
    recon_img.save(recon_path)
    images_to_log.append(wandb.Image(recon_img, caption="Reconstruction"))

    # 마스크
    for k in range(K):
        mask = masks[i, k].numpy()
        mask_path = os.path.join(save_dir, f"{prefix}_epoch{epoch}_mask{k}.png")
        plt.imsave(mask_path, mask, cmap='gray')
        images_to_log.append(wandb.Image(mask_path, caption=f"Mask {k}"))

    wandb.log({f"visualization/epoch_{epoch}": images_to_log}, step=epoch)

# def train_monet(monet, data, optimizer, epoch):
#     monet.train()

#     if use_cuda:
#         data = data.cuda()

#     optimizer.zero_grad()
#     output = monet(data)
#     loss = torch.mean(output['loss'])
#     loss.backward()
#     optimizer.step()

#     wandb.log({"Loss/train_monet": loss.item(), "epoch": epoch})

#     # 시각화를 위한 원본 이미지 추가
#     output['original_image'] = data.detach().cpu()
#     visualize_and_log(output, epoch)

#     return loss.item()

def val_monet(monet, val_data):
    monet.eval()

    if use_cuda:
        val_data = val_data.cuda()

    with torch.no_grad():
        output = monet(val_data)
        loss = torch.mean(output['loss'])

    wandb.log({"Loss/val_monet": loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item()

def train_dae(dae, data, optimizer): #data가 constellation.encode에서 나오는건데, 이거 monet의 output임
    dae.train()

    noise = torch.randn_like(data) * 0.1
    noised_data = data + noise     #noise 제거 학습을 위해 noised 벡터 생성

    if use_cuda:
        data = data.cuda()
        noised_data = noised_data.cuda()

    optimizer.zero_grad()
    recon_batch = dae(noised_data)
    loss = dae.compute_loss(data, recon_batch)
    loss.backward()
    optimizer.step()

    wandb.log({"Loss/train_dae": loss.item(), "epoch": epoch}) #wandb 기록

    return loss.item()


def val_dae(dae, val_data):
    dae.eval()

    torch.manual_seed(42)  # optional: noise 고정
    noise = torch.randn_like(val_data) * 0.1
    noised_data = val_data + noise     #noise 제거 학습을 위해 noised 벡터 생성

    if use_cuda:
        val_data = val_data.cuda()
        noised_data = noised_data.cuda()

    with torch.no_grad():
        recon_batch = dae(noised_data)
        loss = dae.compute_loss(val_data, recon_batch)

    wandb.log({"Loss/val_dae": loss.item(), "epoch": epoch}) #wandb 기록
    
    return loss.item()

def train_bvae(dae, bvae, data, optimizer):
    bvae.train()

    if use_cuda:
        data = data.cuda()

    optimizer.zero_grad()
    recon_data, mu, logvar = bvae(data)
    reconstr_loss, latent_loss = bvae.compute_loss(data, recon_data, mu, logvar, dae, beta=0.5)  # beta 값을 매개변수로 전달, dae 단 사용처
    loss = reconstr_loss + latent_loss
    loss.backward()
    optimizer.step()

    wandb.log({"Loss/train_bvae": loss.item(), "epoch": epoch}) #wandb기록
    
    return loss.item()

def val_bvae(dae, bvae, val_data):
    bvae.eval()

    if use_cuda:
        val_data = val_data.cuda()

    with torch.no_grad():
        recon_data, mu, logvar = bvae(val_data)
        reconstr_loss, latent_loss = bvae.compute_loss(val_data, recon_data, mu, logvar, dae, beta=0.5)  # beta 값을 매개변수로 전달, dae 단 사용처
        loss = reconstr_loss + latent_loss

    wandb.log({"Loss/val_bvae": loss.item(), "epoch": epoch}) #wandb기록

    return loss.item()


def train_scan(bvae, scan, data, label, optimizer):
    scan.eval()

    if use_cuda:
        data, label = data.cuda(), label.cuda()
    optimizer.zero_grad()
    z, y_out, mu, logvar = scan(label) #label forwarding.

    _, x_mu, x_logvar = bvae(data) #image forwarding. bvae단 사용처

    reconstr_loss, latent_loss0, latent_loss1 = scan.compute_loss(data, y_out, label, mu, logvar, x_mu, x_logvar)
    loss = reconstr_loss + latent_loss0 + latent_loss1
    loss.backward()
    torch.nn.utils.clip_grad_norm_(scan.parameters(), max_norm=1.0) #기울기 폭발 막기 위한 클리핑
    optimizer.step()

    wandb.log({"Loss/train_scan": loss.item(), "epoch": epoch}) #wandb기록

    return loss.item()


def val_scan(bvae, scan, val_data, val_label):
    scan.train()

    if use_cuda:
        val_data, val_label = val_data.cuda(), val_label.cuda()

    with torch.no_grad():
        z, y_out, mu, logvar = scan(val_label) #label forwarding.

        _, x_mu, x_logvar = bvae(val_data) #image forwarding. bvae단 사용처

        reconstr_loss, latent_loss0, latent_loss1 = scan.compute_loss(val_data, y_out, val_label, mu, logvar, x_mu, x_logvar)
        loss = reconstr_loss + latent_loss0 + latent_loss1

    wandb.log({"Loss/val_scan": loss.item(), "epoch": epoch}) #wandb기록

    return loss.item()


def train_recomb(dae, bvae, scan, recomb, data, label, in_label, optimizer): #in_label: 개입벡터
    recomb.train()

    factor_sizes = [5, 5, 4, 5, 360] #데이터셋 크기에 맞춰서

    xs = data
    ys0 = label
    ys1 = change_factor_randomly_with_zero_option(ys0, factor_sizes) 
    ys = change_factor_positions(ys0, ys1, factor_sizes)

    if use_cuda:
        xs, ys, ys0, ys1 = xs.cuda(), ys.cuda(), ys0.cuda(), ys1.cuda()

    # 옵티마이저 초기화
    optimizer.zero_grad()

    # 인코딩 과정 
    _,_, mu_0, logvar_0 = scan(ys0) #라벨을 scan으로 :a
    _,_, mu_1, logvar_1 = scan(ys1) #랜덤한 라벨 만들고 scan :b
    _,_, y_mu, y_logvar = scan(ys) #라벨과 랜덤라벨을 합친 라벨을 scan :c
    _, x_mu, x_logvar = bvae(xs) #이미지를 bvae로 

    # Recombinator로 결합
    r_z, r_mu, r_logvar = recomb(mu_0, logvar_0, mu_1, logvar_1)

    # 디코딩 및 손실 계산
    y_out = scan.decoder(r_z)
    symbol_loss = recomb.compute_loss(r_mu, r_logvar, x_mu, x_logvar, y_mu, y_logvar) #a+b의 var,mu가 c의 var,mu와 가까워지게 kl계산, backprop

    # 손실 및 역전파
    loss = symbol_loss  # 심볼 손실만 사용
    loss.backward()
    optimizer.step()

    wandb.log({"Loss/train_recombinator": loss.item(), "epoch": epoch}) #wandb기록

    return loss.item()

def val_recomb(dae, bvae, scan, recomb, val_data, val_label, in_label):
    recomb.eval()

    factor_sizes = [5, 5, 4, 5, 360] #데이터셋 크기에 맞춰서

    xs = val_data
    ys0 = val_label
    ys1 = change_factor_randomly_with_zero_option(ys0, factor_sizes) 
    ys = change_factor_positions(ys0, ys1, factor_sizes)

    if use_cuda:
        xs, ys, ys0, ys1 = xs.cuda(), ys.cuda(), ys0.cuda(), ys1.cuda()

    # 인코딩 과정 
    _,_, mu_0, logvar_0 = scan(ys0) #라벨을 scan으로 :a
    _,_, mu_1, logvar_1 = scan(ys1) #랜덤한 라벨 만들고 scan :b
    _,_, y_mu, y_logvar = scan(ys) #라벨과 랜덤라벨을 합친 라벨을 scan :c
    _, x_mu, x_logvar = bvae(xs) #이미지를 bvae로 

    # Recombinator로 결합
    r_z, r_mu, r_logvar = recomb(mu_0, logvar_0, mu_1, logvar_1)

    # 디코딩 및 손실 계산
    y_out = scan.decoder(r_z)
    symbol_loss = recomb.compute_loss(r_mu, r_logvar, x_mu, x_logvar, y_mu, y_logvar) #a+b의 var,mu가 c의 var,mu와 가까워지게 kl계산, backprop

    # 손실 및 역전파
    loss = symbol_loss  # 심볼 손실만 사용

    wandb.log({"Loss/val_recombinator": loss.item(), "epoch": epoch}) #wandb기록

    return loss.item()

num_classes=379

def process_labels(batch_labels, num_classes): #def process_labels(batch_labels, word_to_idx, num_classes):
    batch_size = batch_labels.size(0)  # 배치 크기 추출
    num_labels = batch_labels.size(1)  # 각 배치의 라벨 개수 추출

    indices = torch.zeros(batch_size, num_labels, dtype=torch.long, device=batch_labels.device)

    # # 각 배치의 라벨을 인덱스로 변환
    # for i in range(batch_size):
    #     for j in range(num_labels):
    #         label = int(batch_labels[i, j].item())  # 라벨 값을 정수형으로 변환
    #         if label in word_to_idx:
    #             indices[i, j] = word_to_idx[label]  # word_to_idx에서 인덱스 가져오기
    #         else:
    #             print(f"Warning: Label {label} not found in word_to_idx.")
    #             indices[i, j] = 0  # 기본값으로 0 추가

    # 멀티핫 인코딩을 위한 빈 벡터 생성 (배치마다 생성)
    one_hot_labels = torch.zeros(batch_size, num_classes, device=indices.device, dtype=torch.float)

    # scatter를 통해 배치별로 멀티핫 인코딩 수행
    one_hot_labels.scatter_(1, indices, 1)

    return one_hot_labels

# word_to_idx = {
#     # x_pos 라벨 (수평 위치)
#     0: -1.0, #x_pos, y_pos 계산 중에 0으로 나와야 하는 라벨이 -1이 되는 가능성이 존재함. 이미 학습된 체크포인트를 어찌 할 수 없어 조치함
#     0: 0.0,  # 왼쪽 (Left)
#     1: 1.0,  # 왼쪽 중앙 (Left-centre)
#     2: 2.0,  # 중앙 (Centre)
#     3: 3.0,  # 오른쪽 중앙 (Right-centre)
#     4: 4.0,  # 오른쪽 (Right)

#     # y_pos 라벨 (수직 위치)
#     5: -1.0, #역시 비슷한 사유
#     5: 0.0,  # 아래 (Bottom)
#     6: 1.0,  # 아래 중앙 (Bottom-middle)
#     7: 2.0,  # 중앙 (Middle)
#     8: 3.0,  # 위쪽 중앙 (Top-middle)
#     9: 4.0,  # 위쪽 (Top)

#     # num_sprites 라벨 (스프라이트 개수)
#     10: 4.0,  # 4개의 스프라이트
#     11: 5.0,  # 5개의 스프라이트
#     12: 6.0,  # 6개의 스프라이트
#     13: 7.0,  # 7개의 스프라이트

#     # curviness 라벨 (곡률)
#     14: 0.0,  # 직선 (Straight)
#     15: 1.0,  # 굽은 선 (Bend)
#     16: 2.0,  # 아치형 (Arch)
#     17: 3.0,  # 말굽형 (Horseshoe)
#     18: 4.0,  # 원형 (Circle)

#     # orientation 라벨 (회전 각도)
#     # 각도 값은 0 ~ 360 범위 내에서 그대로 사용
# }

# # 각도를 위한 word_to_idx에 추가
# for angle in range(0, 360):
#     word_to_idx[angle+19] = float(angle)  # 곡률 19~379까지 그대로 사용

# num_classes = len(word_to_idx) #19+360
num_classes=379

# Transform 적용 
# transforms.Compose(): 여러개의 전처리(transform)을 순차적으로 묶어서 한번에 적용해줌
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())
])

def visualize_constellation(val_x, recon, full_recon, learned_mask, residue, epoch, prefix="constellation_val"):
    import torchvision.utils as vutils
    import math

    i = 0  # 첫 번째 샘플만 시각화

    recon = recon.detach().cpu()
    full_recon = full_recon.detach().cpu()
    learned_mask = learned_mask.detach().cpu()
    residue = residue.detach().cpu()
    original = val_x.detach().cpu()

    images_to_log = []

    # 원본 이미지
    orig_img = transforms.ToPILImage()(original[i])
    images_to_log.append(wandb.Image(orig_img, caption="Original"))

    # Monet 재구성 이미지
    recon_img = transforms.ToPILImage()(recon[i])
    images_to_log.append(wandb.Image(recon_img, caption="Monet Recon"))

    # 전체 재구성 이미지 (LSTM 결과)
    full_recon_img = transforms.ToPILImage()(full_recon[i])
    images_to_log.append(wandb.Image(full_recon_img, caption="Full Recon (LSTM)"))

    # Residue 이미지 (오브젝트 설명 실패 영역)
    residue_img = transforms.ToPILImage()(residue[i])
    images_to_log.append(wandb.Image(residue_img, caption="Residue"))

    # Learned Mask Grid 시각화
    mask_imgs = learned_mask[i].squeeze(0)  # (num_slots, latent_dim)
    mask_imgs = mask_imgs.unsqueeze(1)  # (num_slots, 1, latent_dim, 1)
    mask_imgs = mask_imgs.view(-1, 1, learned_mask.shape[-1], 1)  # reshape for make_grid

    grid = vutils.make_grid(mask_imgs, nrow=int(math.ceil(mask_imgs.shape[0] ** 0.5)), normalize=True, padding=2)
    mask_grid_pil = transforms.ToPILImage()(grid.squeeze(0))  # squeeze for grayscale

    images_to_log.append(wandb.Image(mask_grid_pil, caption="Learned Mask Grid"))

    # wandb 기록
    wandb.log({f"visualization/{prefix}_epoch{epoch}": images_to_log}, step=epoch)



#데이터셋 있으면 있는거 사용, 없으면 make_sprites
dataset = datasets.Sprites(conf.data_dir, mode='train', transform=transform)
val_dataset= datasets.Sprites(conf.data_dir, mode='val', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

monet = models.Monet(conf, height, width).to(device)
monet = nn.DataParallel(monet)

dae = models.DAE(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)
bvae = models.BVAE(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)
scan = models.SCAN(input_size=num_classes, hidden_size=hidden_dim, output_size=r_dim).to(device)
model = models.Constellation(conf, monet, latent_dim, hidden_dim, output_dim, latent_dim, r_dim, height, width).to(device) #input 차원 맞추기
recomb = models.Recombinator(input_size=r_dim, hidden_size=hidden_dim, output_size=r_dim).to(device)

optimizer_monet = optim.Adam(monet.parameters(), lr=1e-3)
#optimizer_monet = optim.RMSprop(monet.parameters(), lr=1e-4)
optimizer_dae = optim.Adam(dae.parameters(), lr=1e-3)
optimizer_bvae = optim.Adam(bvae.parameters(), lr=1e-3)
optimizer_scan = optim.Adam(scan.parameters(), lr=1e-3)
optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
optimizer_recomb = optim.Adam(recomb.parameters(), lr=1e-3)

# monet ckpt파일 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'monet.ckpt')

if os.path.isfile(ckpt_path):
    monet.load_state_dict(torch.load(ckpt_path))
    print('Restored Monet parameters from', ckpt_path)
else:
    print('Start training Monet...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs): #monet 루프
        train_loss=0
        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda()

            # Train MONet
            batch_loss=train_monet(monet, x, optimizer_monet)
            train_loss+=batch_loss

        train_loss/=len(data_loader)

        print(f'MONet Epoch {epoch+1}/{num_epochs}, loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader: 
            val_x, _ = val_batch
            if use_cuda:
                val_x = val_x.cuda()

            batch_loss=val_monet(monet, val_x)
            val_loss+=batch_loss

        val_loss/=len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(monet.state_dict(), ckpt_path)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정

# constellation pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'constellation.ckpt')

if os.path.isfile(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print(f'Restored Constellation parameters from {ckpt_path}')
else: #없으면 train constellation
    print('Start training Constellation...')

    best_val_loss=float('inf')

    for epoch in range(num_epochs): #constellation 루프
        train_loss = 0.0
        num_batches = 0
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

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        wandb.log({"Loss/train_constellation": train_loss, "epoch": epoch}) #wandb기록
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}')

        #constellation -> val
        val_loss=0.0
        num_batches = 0
        for batch in val_data_loader:
            val_x, _ = batch
            if use_cuda:
                val_x = val_x.cuda()

            with torch.no_grad():
                r, mu_q, logvar_q, a, o, learned_mask, recon, residue = model.encode(val_x)

                # Decode
                full_recon, mu, logvar = model.decode(r)

                loss = loss_fn.total_loss(a, full_recon, mu_q, logvar_q, o, learned_mask)
                loss = torch.sum(loss)
                val_loss += loss.item()
                num_batches += 1
        val_loss /= num_batches  # 평균 loss 계산   
                
        wandb.log({"Loss/val_constellation": val_loss, "epoch": epoch}) #wandb기록
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Updated .ckpt, Best Val Loss: {best_val_loss}")

            visualize_constellation(val_x, recon, full_recon, learned_mask, residue, epoch) #시각화

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in model.parameters():
    param.requires_grad = False  # Constellation 모델 고정


# dae pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'dae.ckpt')

if os.path.isfile(ckpt_path):
    dae.load_state_dict(torch.load(ckpt_path))
    print(f'Restored Dae parameters from {ckpt_path}')
else: #없으면 train dae
    print('Start training DAE...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs): #dae 루프
        train_loss=0
        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda()

            # Constellation encode (detach() to avoid computing gradients again)
            r, _, _, _, _, _, _, _ = model.encode(x)

            # Train DAE
            batch_loss=train_dae(dae, r.detach(), optimizer_dae)
            train_loss+=batch_loss

        train_loss/=len(data_loader)

        print(f'Dae Epoch {epoch+1}/{num_epochs}, loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader: 
            val_x, _ = val_batch
            if use_cuda:
                val_x = val_x.cuda()

            r_val, _, _, _, _, _, _, _ = model.encode(val_x)
            batch_loss=val_dae(dae, r_val.detach())
            val_loss+=batch_loss

        val_loss/=len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(dae.state_dict(), ckpt_path)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in model.parameters():
    param.requires_grad = False  # Constellation 모델 고정
for param in dae.parameters():
    param.requires_grad = False  # DAE 모델 고정


# bvae pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'bvae.ckpt')

if os.path.isfile(ckpt_path):
    bvae.load_state_dict(torch.load(ckpt_path)) #['bvae_model_state_dict']
    print(f'Restored Bvae parameters from {ckpt_path}')
else: #없으면 train bvae
    print('Start training BVAE...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs): # bvae 루프
        train_loss=0
        for batch in data_loader:
            x, _ = batch
            if use_cuda:
                x = x.cuda()

            # Constellation encode (detach() to avoid computing gradients again)
            r, _, _, _, _, _, _, _ = model.encode(x)

            # Train BVAE
            batch_loss = train_bvae(dae, bvae, r.detach(), optimizer_bvae)
            train_loss += batch_loss

        train_loss/=len(data_loader)
        
        print(f'Bvae Epoch {epoch+1}/{num_epochs}, train loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader: 
            val_x, _ = val_batch
            if use_cuda:
                val_x = val_x.cuda()

            r_val, _, _, _, _, _, _, _ = model.encode(val_x)
            batch_loss=val_bvae(dae, bvae, r_val.detach())
            val_loss+=batch_loss

        val_loss /= len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss    
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(bvae.state_dict(), ckpt_path)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in model.parameters():
    param.requires_grad = False  # Constellation 모델 고정
for param in dae.parameters():
    param.requires_grad = False  # DAE 모델 고정
for param in bvae.parameters():
    param.requires_grad = False  # bVAE 모델 고정


# scan pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'scan.ckpt')

if os.path.isfile(ckpt_path):
    scan.load_state_dict(torch.load(ckpt_path)) #['scan_model_state_dict'])
    print(f'Restored SCAN parameters from {ckpt_path}')
else: #없으면 train scan
    print('Start training SCAN...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs): #scan 루프
        train_loss=0
        for batch in data_loader:
            x, batch_labels = batch
            one_hot_labels = process_labels(batch_labels, num_classes)
            if use_cuda:
                x = x.cuda()
                one_hot_labels = one_hot_labels.cuda()

            # Constellation encode
            r, _, _, _, _, _, _, _ = model.encode(x)

            # Train SCAN with BVAE encoded results
            batch_loss=train_scan(bvae, scan, r.detach(), one_hot_labels, optimizer_scan)
            train_loss+=batch_loss    

        train_loss/=len(data_loader)

        print(f'Scan Epoch {epoch+1}/{num_epochs}, loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader:
            val_x, val_batch_labels=val_batch
            one_hot_labels = process_labels(val_batch_labels, num_classes)
            if use_cuda:
                val_x = val_x.cuda()
                one_hot_labels = one_hot_labels.cuda()

            # Constellation encode
            r_val, _, _, _, _, _, _, _ = model.encode(val_x)

            # Train SCAN with BVAE encoded results
            batch_loss=val_scan(bvae, scan, r_val.detach(), one_hot_labels)
            val_loss+=batch_loss

        val_loss/=len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(scan.state_dict(), ckpt_path)

for param in monet.parameters():
    param.requires_grad = False  # MONet 모델 고정
for param in model.parameters():
    param.requires_grad = False  # Constellation 모델 고정
for param in dae.parameters():
    param.requires_grad = False  # DAE 모델 고정
for param in bvae.parameters():
    param.requires_grad = False  # bVAE 모델 고정
for param in scan.parameters():
    param.requires_grad = False  # SCAN 모델 고정

# recombinator pt 로드
ckpt_path = os.path.join(conf.checkpoint_dir, 'recomb.ckpt')

if os.path.isfile(ckpt_path):
    recomb.load_state_dict(torch.load(ckpt_path)) #['recombinator_model_state_dict'] 해야될수도 있음
    print(f'Restored Recombinator parameters from {ckpt_path}') 
else: # 없으면 train recombinator
    print('Start training Recombinator...')

    best_val_loss=float('inf')
    for epoch in range(num_epochs):  # 전체 학습 에포크 수
        train_loss=0
        for batch in data_loader:
            x, labels = batch  # 입력 데이터와 라벨 로드
            if use_cuda:
                x = x.cuda()
                labels = labels.cuda()

            r, _, _, _, _, _, _, _ = model.encode(x)

            # 라벨 처리 (현재의 ys0와 개입할 ys1 생성)
            ys0 = process_labels(labels, num_classes)  # 현재 상태 라벨
            ys1 = process_labels(labels, num_classes)  # 개입할 상태 라벨
            # Recombinator 학습 함수 호출
            batch_loss=train_recomb(dae, bvae, scan, recomb, r, ys0, ys1, optimizer_recomb)
            train_loss+=batch_loss

        train_loss/=len(data_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss}')

        val_loss=0.0
        for val_batch in val_data_loader:
            val_x, val_labels = val_batch  # 입력 데이터와 라벨 로드
            if use_cuda:
                val_x = val_x.cuda()
                val_labels = val_labels.cuda()
            r_val, _, _, _, _, _, _, _ = model.encode(val_x)

            # 라벨 처리 (현재의 ys0와 개입할 ys1 생성)
            ys0 = process_labels(val_labels, num_classes)  # 현재 상태 라벨
            ys1 = process_labels(val_labels, num_classes)  # 개입할 상태 라벨
            # Recombinator 학습 함수 호출
            batch_loss=train_recomb(dae, bvae, scan, recomb, r, ys0, ys1, optimizer_recomb)
            val_loss+=batch_loss

        val_loss /=len(val_data_loader)

        if val_loss <best_val_loss:
            best_val_loss=val_loss
            print(f'Updated .ckpt, loss: {best_val_loss}')
            torch.save(recomb.state_dict(), ckpt_path)

print('Train Fin.')