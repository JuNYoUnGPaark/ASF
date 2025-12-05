import os
import time
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------------------------
# 1. UCI-HAR Dataset
# ------------------------------------------------------------------------------
class UCIHARDataset(Dataset):
    def __init__(self, data_path, split='train'):

        self.split = split

        if split == 'train':
            y = np.loadtxt(os.path.join(data_path, 'train', 'y_train.txt'))
            signal_path = os.path.join(data_path, 'train', 'Inertial Signals')
        else:
            y = np.loadtxt(os.path.join(data_path, 'test', 'y_test.txt'))
            signal_path = os.path.join(data_path, 'test', 'Inertial Signals')

        signals = []
        signal_files = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        for signal_file in signal_files:
            filename = os.path.join(signal_path, f'{signal_file}_{split}.txt')
            signal_data = np.loadtxt(filename)
            signals.append(signal_data)

        self.X = np.stack(signals, axis=-1).astype(np.float32)
        self.y = (y - 1).astype(np.int64)

        print(f"{self.X.shape}, {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]


# ------------------------------------------------------------------------------
# 2. ASF Model Components
# ------------------------------------------------------------------------------
class LatentEncoder(nn.Module):
    def __init__(self, input_channels=9, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        s = F.relu(self.bn3(self.conv3(h)))
        s = s.transpose(1, 2)
        return s

class FlowComputer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s):
        B, T, D = s.shape

        flow_raw = s[:, 1:, :] - s[:, :-1, :]
        flow_mag = torch.norm(flow_raw, dim=-1, keepdim=True)
        flow_dir = flow_raw / (flow_mag + 1e-8)

        flow_features = torch.cat(
            [flow_raw, flow_mag.expand(-1, -1, D), flow_dir],
            dim=-1
        )
        return flow_features, flow_raw, flow_mag

class FlowEncoder(nn.Module):
    def __init__(self, flow_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.flow_embed = nn.Linear(flow_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.flow_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.flow_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, flow_features):
        h = self.flow_embed(flow_features)
        h_att, _ = self.attention(h, h, h)

        h_att = h_att.transpose(1, 2)
        h = F.relu(self.bn1(self.flow_conv1(h_att)))
        h = F.relu(self.bn2(self.flow_conv2(h)))

        h_pool = torch.mean(h, dim=-1)
        return h_pool

class StateTransitionPredictor(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s_t):
        B, Tm1, D = s_t.shape
        inp = s_t.reshape(B * Tm1, D)
        out = self.net(inp)
        return out.reshape(B, Tm1, D)

class ASFDCLClassifier(nn.Module):
    def __init__(self,
                 input_channels=9,
                 latent_dim=64,
                 hidden_dim=64,
                 num_classes=6,
                 num_heads=4,
                 projection_dim=128):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.latent_encoder = LatentEncoder(input_channels, latent_dim)
        self.flow_computer = FlowComputer()
        self.flow_encoder = FlowEncoder(latent_dim * 3, hidden_dim, num_heads)
        self.state_predictor = StateTransitionPredictor(latent_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        self.flow_prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_dim)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x, return_details=False):
        s = self.latent_encoder(x)

        s_t = s[:, :-1, :]
        s_next = s[:, 1:, :]
        s_pred_next = self.state_predictor(s_t)

        flow_features, flow_raw, flow_mag = self.flow_computer(s)

        h = self.flow_encoder(flow_features)

        z = self.projection_head(h)
        z = F.normalize(z, dim=-1)

        logits = self.classifier(h)

        if not return_details:
            return logits

        details = {
            "s": s,
            "s_t": s_t,
            "s_next": s_next,
            "s_pred_next": s_pred_next,
            "flow_features": flow_features,
            "flow_raw": flow_raw,
            "flow_mag": flow_mag,
            "h": h,
            "z": z,
            "prototypes": self.flow_prototypes
        }
        return logits, details


# ------------------------------------------------------------------------------
# 3. Loss Functions
# ------------------------------------------------------------------------------
def compute_contrastive_loss(z, labels, temperature=0.07):
    B = z.shape[0]
    device = z.device

    sim_matrix = torch.mm(z, z.t()) / temperature

    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.t()).float()

    positive_mask = positive_mask - torch.eye(B, device=device)

    mask = torch.eye(B, device=device).bool()
    sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))

    exp_sim = torch.exp(sim_matrix_masked)

    pos_sim = (exp_sim * positive_mask).sum(dim=1)

    all_sim = exp_sim.sum(dim=1)

    has_positive = positive_mask.sum(dim=1) > 0

    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = -torch.log(pos_sim[has_positive] / (all_sim[has_positive] + 1e-8))

    return loss.mean()

def compute_asf_dcl_losses(logits, details, labels,
                           lambda_dyn=0.1,
                           lambda_flow=0.05,
                           lambda_proto=0.1,
                           lambda_contrast=0.15,
                           dyn_classes=(0, 1, 2),
                           static_classes=(3, 4, 5),
                           dyn_target=0.7,
                           static_target=0.1,
                           proto_tau=0.1,
                           contrast_temp=0.07):
    device = logits.device

    cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

    s_next = details["s_next"]
    s_pred_next = details["s_pred_next"]
    dyn_loss = F.mse_loss(s_pred_next, s_next)

    flow_mag = details["flow_mag"]
    B, Tm1, _ = flow_mag.shape
    flow_mean = flow_mag.mean(dim=1).view(B)

    dyn_mask = torch.zeros_like(flow_mean, dtype=torch.bool)
    static_mask = torch.zeros_like(flow_mean, dtype=torch.bool)
    for c in dyn_classes:
        dyn_mask = dyn_mask | (labels == c)
    for c in static_classes:
        static_mask = static_mask | (labels == c)

    flow_prior_loss = torch.tensor(0.0, device=device)
    if dyn_mask.any():
        dyn_flow = flow_mean[dyn_mask]
        flow_prior_loss = flow_prior_loss + F.mse_loss(
            dyn_flow, torch.full_like(dyn_flow, dyn_target)
        )
    if static_mask.any():
        static_flow = flow_mean[static_mask]
        flow_prior_loss = flow_prior_loss + F.mse_loss(
            static_flow, torch.full_like(static_flow, static_target)
        )

    h = details["h"]
    prototypes = details["prototypes"]

    h_norm = F.normalize(h, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)

    sim = h_norm @ proto_norm.t()
    proto_logits = sim / proto_tau
    proto_loss = F.cross_entropy(proto_logits, labels, label_smoothing=0.05)

    z = details["z"]
    contrast_loss = compute_contrastive_loss(z, labels, temperature=contrast_temp)

    total_loss = (
        cls_loss +
        lambda_dyn * dyn_loss +
        lambda_flow * flow_prior_loss +
        lambda_proto * proto_loss +
        lambda_contrast * contrast_loss
    )

    loss_dict = {
        "total": total_loss.item(),
        "cls": cls_loss.item(),
        "dyn": dyn_loss.item(),
        "flow_prior": flow_prior_loss.item(),
        "proto": proto_loss.item(),
        "contrast": contrast_loss.item()
    }
    return total_loss, loss_dict