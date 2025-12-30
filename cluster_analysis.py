# ==============================
#  FULL PIPELINE (ANALYSIS + CLUSTERING)
# ==============================
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, f_oneway
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from statsmodels.tsa.seasonal import STL
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
# -------------------------------------------------
# 0. GLOBAL SETTINGS
# -------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs('cluster_analysis', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------
# 1. LOAD RAW DATA
# -------------------------------------------------
DATA_FILE = "train_final.feather"
df = pd.read_feather(DATA_FILE)

# -------------------------------------------------
# 2. SYNTHETIC SERIES (for AE pre-training)
# -------------------------------------------------
N_SYNTHETIC = 5000
SYNTHETIC_LEN = 168

def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05*np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5*np.sin(2*np.pi*np.arange(length)/24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

synthetic_series = generate_synthetic_series(N_SYNTHETIC, SYNTHETIC_LEN)

# -------------------------------------------------
# 3. STL DECOMPOSITION (batch version)
# -------------------------------------------------
def stl_decompose_batch(series, period=24):
    trend_l, season_l, resid_l = [], [], []
    for x in series:
        x = np.nan_to_num(x)
        try:
            res = STL(x, period=period, robust=True).fit()
            trend_l.append(res.trend.astype(np.float32))
            season_l.append(res.seasonal.astype(np.float32))
            resid_l.append(res.resid.astype(np.float32))
        except Exception:
            trend_l.append(np.zeros_like(x))
            season_l.append(np.zeros_like(x))
            resid_l.append(x)
    return np.stack(trend_l), np.stack(season_l), np.stack(resid_l)

trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)

# -------------------------------------------------
# 4. ROW-WISE NORMALISATION
# -------------------------------------------------
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds  = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

trend_syn    = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn    = normalize_rows(resid_syn)

# -------------------------------------------------
# 5. SIMPLE AUTO-ENCODER
# -------------------------------------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        mid = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid), nn.ReLU(),
            nn.Linear(mid, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# -------------------------------------------------
# 6. AE TRAINING (with early stopping)
# -------------------------------------------------
def train_ae(ae, X_train, epochs=10, batch_size=64, lr=1e-3,
             device=DEVICE, verbose=False, patience=3):
    ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best = float('inf')
    wait = 0
    for ep in range(epochs):
        ae.train()
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        running /= len(X_train)

        if verbose:
            print(f"AE ep {ep+1}/{epochs} loss {running:.6f}")

        # early-stop on train loss (no val set)
        if running < best:
            best, wait = running, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: print("Early stop")
                break
    return ae

LATENT_TREND    = 32
LATENT_SEASONAL = 32
LATENT_RESID    = 36          # 100 – (32+32)
LATENT_COVAR    = 16

ae_trend = SimpleAE(SYNTHETIC_LEN, LATENT_TREND)
ae_season = SimpleAE(SYNTHETIC_LEN, LATENT_SEASONAL)
ae_resid = SimpleAE(SYNTHETIC_LEN, LATENT_RESID)

ae_trend = train_ae(ae_trend, trend_syn,    epochs=12, verbose=True)
ae_season = train_ae(ae_season, seasonal_syn, epochs=10, verbose=True)
ae_resid = train_ae(ae_resid, resid_syn,    epochs=8,  verbose=True)

# -------------------------------------------------
# 7. COVARIATES PER BUILDING
# -------------------------------------------------
cov_df = df.groupby('building_id').agg({
    'air_temperature': 'mean',
    'primary_use'    : 'first'
}).reset_index()

ohe = OneHotEncoder(sparse_output=False)
primary_ohe = ohe.fit_transform(cov_df[['primary_use']])
temp_vec    = cov_df[['air_temperature']].values.astype(np.float32)
covariates  = np.concatenate([primary_ohe, temp_vec], axis=1)

ae_cov = SimpleAE(covariates.shape[1], LATENT_COVAR)
ae_cov = train_ae(ae_cov, covariates, epochs=6, verbose=True)

# -------------------------------------------------
# 8. PREPARE REAL SERIES (same length as synthetic)
# -------------------------------------------------
client_ids = df['building_id'].unique()
series_dict = {}
for cid in client_ids:
    s = df[df['building_id']==cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
    s = np.nan_to_num(s, nan=0.0)
    series_dict[cid] = s

resized_series = []
for cid in client_ids:
    s = series_dict[cid]
    if len(s) > SYNTHETIC_LEN:
        resized_series.append(s[:SYNTHETIC_LEN])
    else:
        resized_series.append(np.pad(s, (0, SYNTHETIC_LEN - len(s))))
resized_series = np.array(resized_series)

trend_real, seasonal_real, resid_real = stl_decompose_batch(resized_series)
trend_real    = normalize_rows(trend_real)
seasonal_real = normalize_rows(seasonal_real)
resid_real    = normalize_rows(resid_real)

# -------------------------------------------------
# 9. ENCODE REAL DATA
# -------------------------------------------------
with torch.no_grad():
    Z_tr  = ae_trend.encoder(torch.tensor(trend_real,    dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_se  = ae_season.encoder(torch.tensor(seasonal_real,dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_re  = ae_resid.encoder(torch.tensor(resid_real,    dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_cov = ae_cov.encoder(torch.tensor(covariates,      dtype=torch.float32).to(DEVICE)).cpu().numpy()

encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
encodings_scaled = StandardScaler().fit_transform(encodings)

# -------------------------------------------------
# 2. K-MEANS CLUSTERING
# -------------------------------------------------
K_FORCED = 20
K = min(K_FORCED, len(encodings_scaled))

kmeans = KMeans(
    n_clusters=K,
    n_init=20,
    random_state=SEED
)

labels = kmeans.fit_predict(encodings_scaled)

# Build cluster → client mapping
clusters = {
    f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i] == k]
    for k in range(K)
}

print("\nCluster sizes:")
for k, ids in clusters.items():
    print(f"{k}: {len(ids)} clients")

# -------------------------------------------------
# 3. t-SNE EMBEDDING
# -------------------------------------------------
print("\nComputing t-SNE projection...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=SEED,
    init="pca"
)

embeddings_2d = tsne.fit_transform(encodings_scaled)

# -------------------------------------------------
# 4. DISCRETE COLOR MAP (FIXES FRACTIONAL COLORBAR)
# -------------------------------------------------
cmap = ListedColormap(plt.cm.tab20.colors[:K])
norm = BoundaryNorm(np.arange(-0.5, K + 0.5), K)

# -------------------------------------------------
# 5. t-SNE VISUALIZATION
# -------------------------------------------------
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=labels,
    cmap=cmap,
    norm=norm,
    s=50,
    alpha=0.75,
    edgecolors="k",
    linewidths=0.4
)

plt.title("t-SNE Projection of Client Clusters", fontsize=14, fontweight="bold")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# Force integer cluster IDs on colorbar
cbar = plt.colorbar(scatter, ticks=range(K))
cbar.set_label("Cluster ID")
cbar.set_ticklabels(range(K))

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cluster_analysis/tsne_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✓ Saved: cluster_analysis/tsne_clusters.png")