import os
import time
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
# 1. motion-sense Dataset
# ------------------------------------------------------------------------------
class MotionSenseDataset(Dataset):
    """
    Motion-Sense Dataset Loader
    - 입력: (T, 6) -> (userAcceleration.x, y, z, rotationRate.x, y, z)
    - 라벨: 폴더명(dws, ups, wlk, jog, sit, std)을 파싱하여 인덱스로 매핑
    """

    def __init__(
        self,
        root_dir,
        window_size=128,
        step_size=64,
        normalize=True,
        target_subjects=None,
        scaler=None
    ):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize

        # Motion-Sense의 데이터 폴더 경로 (이미지 기준 A_DeviceMotion_data 폴더)
        self.data_dir = self.root_dir / "A_DeviceMotion_data"
        
        # 1) 데이터 로드 및 통합
        df_all = self._load_all_data()

        if target_subjects is not None:
            df_all = df_all[df_all['subject_id'].isin(target_subjects)].copy()
            print(f"Dataset initialized with subjects: {target_subjects}")
            print(f"Total rows after filtering: {len(df_all)}")
            
        # 2) 라벨 -> 인덱스 매핑
        # Motion-Sense의 6개 클래스: dws, ups, wlk, jog, sit, std
        activities = ['dws', 'jog', 'sit', 'std', 'ups', 'wlk']
        self.label2idx = {label: i for i, label in enumerate(activities)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        df_all["label_idx"] = df_all["activity"].map(self.label2idx)

        # 3) 정규화 (StandardScaler)
        # MotionSense 컬럼: userAcceleration.x/y/z (acc), rotationRate.x/y/z (gyro)
        feat_cols = [
            "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
            "rotationRate.x", "rotationRate.y", "rotationRate.z"
        ]
        feats = df_all[feat_cols].values.astype(np.float32)

        if self.normalize:
            if scaler is None:
                # 스케일러가 없으면(Train용) -> 새로 맞춤(fit)
                self.scaler = StandardScaler()
                feats = self.scaler.fit_transform(feats)
            else:
                # 스케일러가 있으면(Test용) -> 기존 것 사용(transform)
                self.scaler = scaler
                feats = self.scaler.transform(feats)
        else:
            self.scaler = None
        
        df_all[feat_cols] = feats

        # 4) 슬라이딩 윈도우 생성 (Subject, Activity, Trial 별로 그룹화)
        X_list = []
        y_list = []

        # trial_id는 각 csv 파일을 구분하기 위해 _load_all_data에서 생성해야 함
        for _, g in df_all.groupby(["subject_id", "activity", "trial_id"]):
            g = g.sort_values("timestamp_idx").reset_index(drop=True)
            
            data = g[feat_cols].values 
            labels = g["label_idx"].values
            n = len(g)

            if n < window_size:
                continue

            for start in range(0, n - window_size + 1, step_size):
                end = start + window_size
                w_data = data[start:end]
                w_labels = labels[start:end]

                # 윈도우 라벨 (Mode)
                majority_label = np.bincount(w_labels).argmax()

                X_list.append(w_data.astype(np.float32))
                y_list.append(majority_label)

        self.X = np.stack(X_list) if len(X_list) > 0 else np.zeros((0, window_size, 6), dtype=np.float32)
        self.y = np.array(y_list, dtype=np.int64)

        print(f"[MotionSenseDataset] windows: {len(self.X)}, classes: {len(self.label2idx)}")
        print(f"Classes map: {self.label2idx}")

    def _load_all_data(self):
        """
        A_DeviceMotion_data 내부의 모든 폴더를 순회하며 CSV 로드
        """
        all_dfs = []
        
        # data_dir 내부의 폴더들 (예: dws_1, jog_9 ...)
        if not self.data_dir.exists():
             raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        for folder in os.listdir(self.data_dir):
            folder_path = self.data_dir / folder
            if not folder_path.is_dir():
                continue
            
            # 폴더명 파싱 (예: dws_1 -> activity=dws, subject=1)
            parts = folder.split('_')
            activity_label = parts[0]
            subject_id = parts[1]

            # 폴더 내 csv 파일 읽기 (보통 sub_1.csv 같은 형태)
            for csv_file in os.listdir(folder_path):
                if not csv_file.endswith(".csv"):
                    continue
                
                file_path = folder_path / csv_file
                df = pd.read_csv(file_path)
                
                # Unnamed: 0 컬럼이 타임스탬프 역할(인덱스)
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "timestamp_idx"})
                else:
                    df["timestamp_idx"] = range(len(df))

                df["activity"] = activity_label
                df["subject_id"] = int(subject_id)
                df["trial_id"] = folder  # 폴더명 자체를 trial 식별자로 사용

                all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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
                 input_channels=6,
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
                           dyn_classes=(0, 1, 4, 5),
                           static_classes=(2, 3),
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