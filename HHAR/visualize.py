import io
import os
import time
import copy
import torch
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from model_utils import HHARPhoneIMUDataset, ASFDCLClassifier, seed_worker
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: 'fvcore' not installed. FLOPs calculation will be skipped.")

ACTIVITY_LABELS = ['Biking', 'Sitting', 'StairsDown', 'StairsUp', 'Standing', 'Walking']
ACTIVITY_LABELS_CM = ACTIVITY_LABELS
n_labels = len(ACTIVITY_LABELS)
colors = sns.color_palette("hsv", n_colors=n_labels)
ACTIVITY_COLOR_MAP = dict(zip(ACTIVITY_LABELS, colors))
# ------------------------------------------------------------------------------
# 1. Visualization Functions
# ------------------------------------------------------------------------------
def plot_classification_results(y_true, y_pred, save_path=None):
    print("\n" + "="*80)
    print("Classification Report")
    print("="*80)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=ACTIVITY_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', square=True,
                annot_kws={"size": 22}, cbar_kws={"shrink": 0.7},
                xticklabels=ACTIVITY_LABELS_CM, yticklabels=ACTIVITY_LABELS_CM)
    plt.xlabel('Predicted Label', fontsize=12.5)
    plt.ylabel('True Label', fontsize=11)
    plt.title('')
    plt.xticks(rotation=90, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600)

    plt.show()
    plt.close()

def _extract_featvec_before_classifier(model, xb, device):
    xb = xb.to(device)
    _, details = model(xb, return_details=True)
    feat_vec = details["h"]  
    return feat_vec.detach().cpu()

def plot_tsne_from_cached_features(feats, labels, save_path=None, max_points=2000):
    all_features = feats
    all_labels = labels 

    N = all_features.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)

    X_sel = all_features[idx] 
    y_sel = all_labels[idx] 

    effective_perp = min(30, len(X_sel) - 1)
    effective_perp = max(effective_perp, 5)

    print(f"\nRunning t-SNE on {len(X_sel)} points (Perplexity={effective_perp})...")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=effective_perp,
        max_iter=2000,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(X_sel)  

    df = pd.DataFrame(proj, columns=["Dim1", "Dim2"])
    df["label"] = [ACTIVITY_LABELS[l] for l in y_sel]

    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(
        data=df,
        x="Dim1",
        y="Dim2",
        hue="label",
        palette=ACTIVITY_COLOR_MAP,
        hue_order=ACTIVITY_LABELS,
        legend="full",
        alpha=0.8,
        s=40,
    )
    plt.title("t-SNE of embeddings extracted by the model", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.legend(
        loc="upper right",
        fontsize=13,
        labelspacing=0.2,
        markerscale=1.5
    )
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()

def visualize_tsne_raw(dataloader, save_path=None, max_points=2000):
    all_raw = []
    all_labels = []

    for xb, yb in dataloader:
        all_raw.append(xb.cpu().numpy())   
        all_labels.append(yb.cpu().numpy())  

    all_raw = np.concatenate(all_raw, axis=0)      
    all_labels = np.concatenate(all_labels, axis=0) 

    N = all_raw.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)

    X_sel = all_raw[idx]
    y_sel = all_labels[idx] 

    X_flat = X_sel.reshape(X_sel.shape[0], -1)

    effective_perp = min(30, len(X_flat) - 1)
    effective_perp = max(effective_perp, 5)

    print(f"\nRunning Raw Data t-SNE on {len(X_sel)} points...")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=effective_perp,
        max_iter=2000,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(X_flat)

    df = pd.DataFrame(proj, columns=["Dim1", "Dim2"])
    df["label"] = [ACTIVITY_LABELS[l] for l in y_sel]

    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(
        data=df,
        x="Dim1",
        y="Dim2",
        hue="label",
        palette=ACTIVITY_COLOR_MAP,
        hue_order=ACTIVITY_LABELS,
        legend="full",
        alpha=0.8,
        s=40, 
    )
    plt.title("t-SNE of raw data before model processing", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.legend(
        loc="upper right",
        fontsize=13,
        labelspacing=0.2,
        markerscale=1.5
    )
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()

def measure_efficiency(model, device, input_shape=(1, 80, 3), warmup=10, iters=100):
    """
    모델의 파라미터 수, FLOPs, 추론 속도를 측정합니다.
    """
    measure_device = torch.device('cpu')
    model_cpu = copy.deepcopy(model).to(measure_device)
    model_cpu.eval()
    
    real_input_shape = list(input_shape)
    real_input_shape[0] = 1
    # 더미 입력 데이터 생성
    sample_input = torch.randn(tuple(real_input_shape)).to(measure_device)

    # -------------------------------------------------
    # 1) 파라미터 수
    # -------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6  # million params

    # -------------------------------------------------
    # 2) FLOPs 측정 (fvcore 사용 가능할 때만)
    # -------------------------------------------------
    flops_m = None
    if FVCORE_AVAILABLE:
        try:
            with torch.no_grad():
                fake_out = io.StringIO()
                fake_err = io.StringIO()
                with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_err):
                    flops = FlopCountAnalysis(model_cpu, (sample_input,))
                    total_flops = flops.total()
            flops_m = total_flops / 1e6  # to millions
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            flops_m = None

    # -------------------------------------------------
    # 3) 추론 시간 측정
    # -------------------------------------------------
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model_cpu(sample_input)

        start = time.time()
        for _ in range(iters):
            _ = model_cpu(sample_input)
        end = time.time()

    avg_sec = (end - start) / iters
    inference_ms = avg_sec * 1000.0

    del model_cpu

    return {
        "params_m": params_m,
        "flops_m": flops_m,
        "inference_ms": inference_ms,
    }
# ------------------------------------------------------------------------------
# 2. Main 
# ------------------------------------------------------------------------------
def main():
    SEED = 42

    DATA_PATH = '/content/HHARPhone'
    MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/ASF/HHAR/HHAR_ASF.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("-" * 80)
    print("Running Visualization...")
    print(f"Loading weights from: {MODEL_PATH}")
    print("-" * 80)

    full_dataset = HHARPhoneIMUDataset(DATA_PATH, window_size=128, step_size=64, fs=50)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    if hasattr(full_dataset, 'idx2label'):
        target_names = [full_dataset.idx2label[i] for i in range(len(full_dataset.idx2label))]
        print(f"Dataset Labels Detected: {target_names}")
    else:
        target_names = None
        print("Warning: idx2label not found.")

    g_split = torch.Generator().manual_seed(SEED)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=g_split)

    g = torch.Generator()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                             num_workers=2, worker_init_fn=seed_worker, generator=g)

    model = ASFDCLClassifier(input_channels=6, num_classes=6).to(DEVICE)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model.eval()
    
    # 모델 효율성 측정 및 출력
    print("-" * 80)
    print("Evaluating Model Efficiency...")
    metrics = measure_efficiency(model, DEVICE, input_shape=(1, 128, 6))
    
    p_m = metrics['params_m']
    f_m = metrics['flops_m']
    t_ms = metrics['inference_ms']

    print(f"1. Parameters       : {p_m:.4f} M")
    
    if f_m is not None:
        print(f"2. FLOPs / sample   : {f_m:.3f} M")
    else:
        print(f"2. FLOPs / sample   : N/A (Install fvcore)")
        
    print(f"3. Infer time       : {t_ms:.2f} ms/sample")
    print("-" * 80)

    all_preds = []
    all_labels = []
    all_features = []

    print("Extracting features and predictions...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            feats = _extract_featvec_before_classifier(model, x, DEVICE)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_features.append(feats.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.concatenate(all_features, axis=0)

    # --------------------------------------------------------------------------
    # 3. Visualize
    # --------------------------------------------------------------------------
    save_dir = os.path.dirname(MODEL_PATH)

    # (1) Confusion Matrix
    print("Generating Confusion Matrix...")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_classification_results(all_labels, all_preds, save_path=cm_path)

    # (2) Learned Features t-SNE (Model Output)
    print("Generating Feature t-SNE...")
    tsne_feat_path = os.path.join(save_dir, "tsne_features.png")
    plot_tsne_from_cached_features(all_features, all_labels, save_path=tsne_feat_path)

    # (3) Raw Data t-SNE (Input Data)
    print("Generating Raw Data t-SNE...")
    tsne_raw_path = os.path.join(save_dir, "tsne_raw.png")
    visualize_tsne_raw(test_loader, save_path=tsne_raw_path)

    print("\nVisualization Complete!")

if __name__ == "__main__":
    main()