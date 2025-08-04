# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
import torchvision

from torch_geometric.nn import GCNConv
from scipy.optimize import linear_sum_assignment
import itertools

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#GNN
class RelationalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, r_dim, num_slots):
        super(RelationalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, r_dim)
        self.fc_logvar = nn.Linear(hidden_dim, r_dim)
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.global_mlp = nn.Sequential(
            nn.Linear(num_slots * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        ).to(device)


    def forward(self, x, edge_index):
        batch_size, num_slots, features = x.shape

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = x.view(batch_size, num_slots, self.hidden_dim)

        x = x.view(batch_size, -1)
        global_info = self.global_mlp(x)

        mu_q = self.fc_mu(global_info)
        logvar_q = self.fc_logvar(global_info)
        return mu_q, logvar_q

# LSTM
class SequentialLSTM(nn.Module):
    def __init__(self, r_dim, hidden_dim, latent_dim, num_slots):
        super(SequentialLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(r_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mu 추출
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # logvar 추출
        self.num_slots = num_slots

    def forward(self, r): #r: representation
        batch_size, r_dim = r.size()
        h_t, c_t = self.init_hidden(batch_size, r.device)

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
    def __init__(self, conf, monet, input_dim, hidden_dim, output_dim, latent_dim, r_dim, height, width):
        super(Constellation, self).__init__()
        self.monet = monet
        self.gnn = RelationalGNN(latent_dim, hidden_dim, r_dim, conf.num_slots)
        self.lstm = SequentialLSTM(r_dim, hidden_dim, latent_dim, conf.num_slots)
        # gpt는 추가 loss function 없이 알아서 위치 추출을 하도록 학습된다고 주장하지만 확인 필요할 것으로 보임
        self.mask_extractor = nn.Sequential(
            nn.Conv1d(conf.num_slots, 16, 3, padding=1), #차원 이슈 해결
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.Softmax(dim=2)
        )    
    
    def encode(self, x):
        monet_output = self.monet(x)
        masks = monet_output['masks']
        recon = monet_output['reconstructions']
        o = monet_output['zs']
        batch_size = o.shape[0]
        
        learned_mask = self.mask_extractor(o) #.............monet_output의 latent vector만 가지고 mask extractor가 된다고,,?

        a = o * learned_mask
        
        residue = o * (1 - learned_mask)
        num_nodes = o.shape[1]
        edge_index = self.create_fully_connected_edge_index(num_nodes) 

        # print("a.device: ", a.device)
        # print("edge_index.device: ", edge_index.device)
        mu_q, logvar_q = self.gnn(a, edge_index)
        r = self.lstm.reparameterize(mu_q, logvar_q)
        return r, mu_q, logvar_q, a, o, learned_mask, recon, residue

    def decode(self, r):
        mu_outputs, logvar_outputs = self.lstm(r)
        recon = self.lstm.reparameterize(mu_outputs, logvar_outputs)
        return recon, mu_outputs, logvar_outputs

    #완전 연결 edge index 생성 함수.
    def create_fully_connected_edge_index(self, num_nodes):
        edge_index = list(itertools.permutations(range(num_nodes), 2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

class LossFunctions(nn.Module):
    def __init__(self, conf, beta=4, gamma_init=0.1, latent_dim=16):
        super(LossFunctions, self).__init__()
        self.beta = beta
        self.gamma = nn.Parameter(torch.tensor([gamma_init] * conf.num_slots * latent_dim))

    # 논문에 나온 손실 함수 구현
    def reconstruction_loss(self, ai, a_hat):
        loss = 0.0
        batch_size, num_slots, latent_dim = ai.size()

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
        return loss

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
        return L_total