# License: MIT
# Author: Karl Stelzner

import os
import sys

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.random import random_integers
from PIL import Image
import torchvision.transforms as transforms
from spriteworld import sprite, renderers

import config
conf=config.sprite_config

# Define the possible properties
shapes = ['square', 'circle', 'triangle']
sizes = [0.08, 0.09]
orientations_square = list(np.linspace(0, 90, num=6))
orientations_triangle = list(np.linspace(0, 120, num=8))

# Function to generate a random sprite at a specific position
def random_sprite_at_position(x, y):
    color = np.random.choice(256, 3)
    shape = np.random.choice(shapes)
    size = np.random.choice(sizes)
    if shape == 'square':
        orientations = orientations_square
    elif shape == 'triangle':
        orientations = orientations_triangle
    else : orientations = [0]
    orientation = np.random.choice(orientations)

    return sprite.Sprite(
            shape=shape,
            x=x,
            y=y,
            scale=size,
            c0=color[0],
            c1=color[1],
            c2=color[2],
            angle=orientation
        )

# Function to generate positions along a line or curve
def generate_positions(num_sprites, curviness, center, orientation):
    structure_length = 0.65 + 0.05 * (num_sprites-4)
    if curviness == 0:
        # Straight line
        x = np.zeros(num_sprites)
        y = np.linspace(-structure_length/2, structure_length/2, num_sprites)
    else:
        # Curved line (circle for large curviness)
        radius = 2 / (np.pi * curviness)
        theta = structure_length / radius
        angle_range = np.linspace(-theta/2, theta/2, num_sprites)
        x = radius * np.cos(angle_range) - radius
        y = radius * np.sin(angle_range)

    # Rotate positions by orientation angle
    cos_theta, sin_theta = np.cos((orientation-90) * np.pi/180), np.sin((orientation-90) * np.pi/180)
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y

    # Translate to center position
    x_trans = center[0] + x_rot
    y_trans = center[1] + y_rot

    return x_trans, y_trans

# Function to check if all positions are within the bounds
def positions_within_bounds(x_positions, y_positions, margin=0.1):
    return (x_positions >= margin).all() and (x_positions <= 1 - margin).all() and (y_positions >= margin).all() and (y_positions <= 1 - margin).all()


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def make_sprites(n=conf.data_num, height=128, width=128): #h,w=64 였음 #500000?? 200000?? 100000
    renderer = renderers.PILRenderer(image_size=(height, width),anti_aliasing=1)
    images = np.zeros((int(n), int(height), int(width), 3))
    labels = np.zeros((n, 5))
    print('Generating sprite dataset...')
    for i in range(n):
        while True:
            num_sprites = np.random.randint(4, 7) #4,5,6,7
            curviness = np.random.randint(5) # 0:Straight 1:Bend 2:Arch 3:Horseshoe 4:Circle
            center = np.random.uniform(0.2, 0.8, size=2)
            if curviness == 0:
                orientation = np.random.choice(range(0, 180, 5))
            else: orientation = np.random.choice(range(0, 360, 5))
            x_positions, y_positions = generate_positions(num_sprites, curviness, center, orientation)
            if positions_within_bounds(x_positions, y_positions):
                break
        sprites = [random_sprite_at_position(x, y) for x, y in zip(x_positions, y_positions)]

        # Generate labels for x,y position
        x_pos = np.clip(((np.mean(x_positions)-0.2)/0.6)//0.2, 0, 4)  # 0~4 범위로 제한
        y_pos = np.clip(((np.mean(y_positions)-0.2)/0.6)//0.2, 0, 4)  # 0~4 범위로 제한 #라벨로 -1이 자꾸 나오는 이슈 해결

        images[i], labels[i] = renderer.render(sprites) / 255.0, [x_pos, y_pos+5, num_sprites+10-4, curviness+14, orientation+19]
        if i % 100 == 0:
            progress_bar(i, n)
    images = np.clip(images, 0.0, 1.0)

    return {'x_train': images[:8 * n // 10], #80%
            'labels_train': labels[:8 * n // 10],
            'x_val': images[8 * n // 10: 9 * n // 10], #10%
            'labels_val': labels[8 * n // 10: 9 * n // 10],
            'x_test': images[8 * n // 10:], #10%
            'labels_test': labels[8 * n // 10:]}

class Sprites(Dataset):
    def __init__(self, directory, mode, n=conf.data_num, canvas_size=128, #canvas_size를 64에서 128로
                transform=None): #mode=train/val/test       #원래는 train=True
        np_file = 'sprites_{}_{}.npz'.format(n, canvas_size)
        full_path = os.path.join(directory, np_file)
        os.makedirs(directory, exist_ok=True)
        if not os.path.isfile(full_path):
            gen_data = make_sprites(n, canvas_size, canvas_size)
            np.savez(full_path, **gen_data) #경로통일 full_path로!

        data = np.load(full_path)

        self.transform = transform

        if mode=='train':
            self.images = data['x_train']
            self.labels = data['labels_train']
        elif mode=='val':
            self.images = data['x_val']
            self.labels = data['labels_val']
        else:
            self.images = data['x_test']
            self.labels = data['labels_test']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]