# License: MIT
# Author: Karl Stelzner

#dataset visualize하는 코드

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import os

import models
import dataset
import config

import matplotlib.pyplot as plt


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#데이터셋 없으면 생성하는 코드
trainset = dataset.Sprites('./data', mode='test') 

trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=128,
                                            shuffle=True, num_workers=2)

# def numpify(tensor):
#     return tensor.cpu().detach().numpy()

def numpify(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy()
    else:
        # 이미 numpy array면 그냥 반환
        return tensor_or_array


def visualize_masks(images, masks, labels,  save_dir='./vis', prefix='test'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(8): #이미지 8개 저장한다는 뜻(0~7번 이미지)
        fig, axs = plt.subplots(2, 9, figsize=(9,2))

        # 1행: 원본, 전체 마스크 합
        axs[0, 0].imshow(images[i]) #.transpose(1, 2, 0))
        axs[0, 0].set_title('Input')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(masks[i].sum(axis=0),cmap='viridis')
        axs[0, 1].set_title('Mask(Sum)')
        axs[0, 1].axis('off')
        axs[0, 3].text(5, 10, f"Label: {labels[i]}", color='white',
                fontsize=9, backgroundcolor='black')

        # 1행 [2:] 칸에 라벨 텍스트를 중앙에 출력
        label_text = f'Label: {labels[i]}'
        for col in range(2, 9):
            axs[0, col].axis('off')
        # axs[0, 6].text(
        #     1.0, 0.5, label_text,
        #     transform=axs[0, 2].transAxes,
        #     fontsize=10,
        #     color='black',
        #     ha='right',
        #     va='center',
        #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        # )

        # 2행
        for slot_idx in range(8): #8=num slots
            slot_mask = masks[i][slot_idx] #.sum(axis=0)  # 채널 축 합쳐서 (H,W)로
            print("slot_masks_size",len(slot_mask))
            print("slot_masks_size",len(slot_mask[0]))
            print("slot_masks_size",len(slot_mask[0][0]))

            axs[1, slot_idx].imshow(slot_mask)
            axs[1, slot_idx].set_title(f'Mask {slot_idx}')
            axs[1, slot_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{prefix}_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")


for i, data in enumerate(trainloader, 0):
    images, labels, masks = data
    images = images.cpu().to(torch.float32)
    # images = np.clip(images, 0.0, 1.0) 
    images = np.clip(images.detach().numpy(), 0.0, 1.0)

    if i <10:
        #masks_list_npy = [numpify(slot) for slot in masks]
        masks_list_npy = [numpify(slot).astype(np.float32) for slot in masks]
        print("masks list numpy shape",len(masks_list_npy)) #128
        print("masks list numpy shape",len(masks_list_npy[0])) #8
        print("masks list numpy shape",len(masks_list_npy[0][0])) #128
        print("masks list numpy shape",len(masks_list_npy[0][0][0])) #128
        print("masks list numpy shape",len(masks_list_npy[0][0][0][0])) #3
        visualize_masks(numpify(images),
                        masks_list_npy,
                        labels,
                        save_dir='./vis', prefix='test'
                        )