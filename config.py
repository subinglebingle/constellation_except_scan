# License: MIT
# Author: Karl Stelzner

from collections import namedtuple
import os

config_options = [
    'data_num', #n=100000
    # Training config
    'vis_every',  # Visualize progress every X iterations
    'batch_size',
    'num_epochs',
    'load_parameters',  # Load parameters from checkpoint
    'checkpoint_file',  # File for loading/storing checkpoints
    'checkpoint_dir', #내가 추가... checkpoints들 저장할 파일
    'data_dir',  # Directory for the training data
    'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blocks in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
    'beta', #내가 추가
    'gamma', #내가 추가
    'vae_sigma', #내가 추가
    'vae_beta', #내가 추가
    'vae_gamma', #내가 추가
    'latent_dim', #내가추가
    'r_dim', #내가 추가 #relatinoal latents인 r의 차원 결정
    'hidden_dim' #내가 추가 #lstm
]

MonetConfig = namedtuple('MonetConfig', config_options)

sprite_config = MonetConfig(data_num=1500, #n=100000 #test할때 1500장으로 하는거 추천
                            vis_every=50, #몇개마다 visualization할건지
                            batch_size=64, #논문: 64
                            num_epochs=20, #논문에서는 모든 실험 1000000 iteration 이상 진행
                            load_parameters=True,
                            checkpoint_file='./checkpoints/monet.ckpt', #나중에 없애는게 목표
                            checkpoint_dir='./checkpoints',
                            data_dir='./data/',
                            parallel=True,
                            num_slots=8, #constellation dataset에 맞춰서 늘림
                            num_blocks=5,
                            channel_base=128, #64
                            bg_sigma=0.09, #background sigma, 1번째 슬롯에만 쓰이는 파라미터, 잘 조절하면 slot1에 배경색만 나오게된다. (논문: 0.09)
                            fg_sigma=0.11, #논문: 0.11
                            beta=8.0, #monet 논문 0.5 #인겸님 코드 8.0
                            gamma=0.5, #monet 논문 0.5
                            vae_sigma=0.05, #논문 0.05  #(?)
                            vae_beta=0.5, #논문 0.5  #(?)
                            vae_gamma=0.25, #논문 0.25  #(?)
                            latent_dim=16,
                            r_dim=16, #논문에 안나와서 임의로 정함. 수정 가능
                            hidden_dim=128 #논문에 안나왔었던거같음. 확인필요, 수정가능
                           )
