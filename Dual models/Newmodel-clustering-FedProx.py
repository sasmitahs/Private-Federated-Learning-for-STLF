import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, f_oneway
import warnings
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.tsa.seasonal import STL
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('cluster_analysis', exist_ok=True)

# -----------------------------
# PARAMETERS
# -----------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
LATENT_DIM_TOTAL = 100
LATENT_TREND = 32
LATENT_SEASONAL = 32
LATENT_RESID = LATENT_DIM_TOTAL - (LATENT_TREND + LATENT_SEASONAL)
LATENT_COVAR = 16
AE_EPOCHS_TREND = 12
AE_EPOCHS_SEASON = 10
AE_EPOCHS_RESID = 8
AE_EPOCHS_COVAR = 6
AE_BATCH = 64
AE_LR = 1e-3
PATIENCE = 3  # early stopping patience
K_FORCED = 20
synthetic_length = 168

# Load real data
df = pd.read_feather("train_final.feather")

# -----------------------------
# Synthetic time series generation
# -----------------------------
n_synthetic = 5000
def generate_synthetic_series(n, length):
    series = []
    for _ in range(n):
        trend = 0.05 * np.arange(length) + np.random.normal(0, 0.1, length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(length) / 24) + np.random.normal(0, 0.05, length)
        resid = np.random.normal(0, 0.1, length)
        series.append(trend + seasonal + resid)
    return np.array(series, dtype=np.float32)

synthetic_series = generate_synthetic_series(n_synthetic, synthetic_length)

# -----------------------------
# STL decomposition
# -----------------------------
def stl_decompose_batch(series, period=24):
    trend_list, seasonal_list, resid_list = [], [], []
    for x in series:
        x = np.nan_to_num(x)
        try:
            stl = STL(x, period=period, robust=True)
            res = stl.fit()
            trend_list.append(res.trend.astype(np.float32))
            seasonal_list.append(res.seasonal.astype(np.float32))
            resid_list.append(res.resid.astype(np.float32))
        except:
            trend_list.append(x * 0)
            seasonal_list.append(x * 0)
            resid_list.append(x)
    return np.stack(trend_list), np.stack(seasonal_list), np.stack(resid_list)

trend_syn, seasonal_syn, resid_syn = stl_decompose_batch(synthetic_series)

# -----------------------------
# Normalization
# -----------------------------
def normalize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds

trend_syn = normalize_rows(trend_syn)
seasonal_syn = normalize_rows(seasonal_syn)
resid_syn = normalize_rows(resid_syn)

# -----------------------------
# Simple AE class
# -----------------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        mid = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# -----------------------------
# AE Training with Early Stopping
# -----------------------------
def train_ae(ae, X_train, X_val=None, epochs=10, batch_size=64, lr=1e-3, device=DEVICE, verbose=False, patience=3):
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    best_val_loss = float('inf')
    wait = 0
    for ep in range(epochs):
        # Training
        ae.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        running /= len(X_train)
        # Validation
        val_loss = None
        if X_val is not None:
            ae.eval()
            with torch.no_grad():
                recon_val = ae(X_val_t)
                val_loss = loss_fn(recon_val, X_val_t).item()
        if verbose:
            print(f"AE epoch {ep+1}/{epochs} train_loss={running:.6f} val_loss={val_loss:.6f}")
        # Early stopping
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {ep+1}")
            break
    ae.eval()
    return ae

# -----------------------------
# Train AEs for time series components
# -----------------------------
ae_trend = SimpleAE(synthetic_length, LATENT_TREND)
ae_season = SimpleAE(synthetic_length, LATENT_SEASONAL)
ae_resid = SimpleAE(synthetic_length, LATENT_RESID)
ae_trend = train_ae(ae_trend, trend_syn, epochs=AE_EPOCHS_TREND, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_season = train_ae(ae_season, seasonal_syn, epochs=AE_EPOCHS_SEASON, batch_size=AE_BATCH, lr=AE_LR, verbose=True)
ae_resid = train_ae(ae_resid, resid_syn, epochs=AE_EPOCHS_RESID, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

# -----------------------------
# Covariates AE (per building)
# -----------------------------
# Aggregate numeric and categorical covariates per building
cov_df = df.groupby('building_id').agg({
    'air_temperature': 'mean',
    'primary_use': 'first'
}).reset_index()
primary_use_ohe = OneHotEncoder(sparse_output=False)
primary_use_encoded = primary_use_ohe.fit_transform(cov_df[['primary_use']])
air_temp = cov_df[['air_temperature']].values.astype(np.float32)
covariates_per_building = np.concatenate([primary_use_encoded, air_temp], axis=1)
ae_covariates = SimpleAE(covariates_per_building.shape[1], LATENT_COVAR)
ae_covariates = train_ae(ae_covariates, covariates_per_building, epochs=AE_EPOCHS_COVAR, batch_size=AE_BATCH, lr=AE_LR, verbose=True)

# -----------------------------
# Prepare real series for encoding
# -----------------------------
client_ids = df['building_id'].unique()
series_dict = {}
for cid in client_ids:
    series = df[df['building_id'] == cid].sort_values('timestamp')['meter_reading'].values.astype(np.float32)
    series = np.nan_to_num(series, nan=0.0)
    series_dict[cid] = series
resized_series = []
for cid in client_ids:
    s = series_dict[cid]
    if len(s) > synthetic_length:
        resized_series.append(s[:synthetic_length])
    else:
        pad_len = synthetic_length - len(s)
        resized_series.append(np.pad(s, (0, pad_len)))
resized_series = np.array(resized_series)
trend_real, seasonal_real, resid_real = stl_decompose_batch(resized_series)
trend_real = normalize_rows(trend_real)
seasonal_real = normalize_rows(seasonal_real)
resid_real = normalize_rows(resid_real)

# -----------------------------
# Encode all embeddings
# -----------------------------
with torch.no_grad():
    Z_tr = ae_trend.encoder(torch.tensor(trend_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_se = ae_season.encoder(torch.tensor(seasonal_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_re = ae_resid.encoder(torch.tensor(resid_real, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    Z_cov = ae_covariates.encoder(torch.tensor(covariates_per_building, dtype=torch.float32).to(DEVICE)).cpu().numpy()
# Concatenate embeddings per building
encodings = np.concatenate([Z_tr, Z_se, Z_re, Z_cov], axis=1)
encodings_scaled = StandardScaler().fit_transform(encodings)

# -----------------------------
# KMeans clustering
# -----------------------------
K = min(K_FORCED, len(client_ids))
km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
labels = km.fit_predict(encodings_scaled)
clusters = {f"cluster_{k}": [client_ids[i] for i in range(len(client_ids)) if labels[i] == k] for k in range(K)}
for name, ids in clusters.items():
    print(f"{name}: {len(ids)} buildings; sample: {ids[:5]}")

# ============================================================================
# CLUSTER ANALYSIS FUNCTIONS
# ============================================================================
# (Paste all the analysis functions here exactly as in the previous corrected cluster_analysis.py)
# For brevity, assuming they are defined as before: analyze_cluster_composition, plot_cluster_timeseries, etc.
# up to run_complete_cluster_analysis

def analyze_cluster_composition(clusters, cov_df):
    """Analyze building types and characteristics per cluster"""
    print("\n" + "="*80)
    print("CLUSTER COMPOSITION ANALYSIS")
    print("="*80)
    composition_data = []
    for cluster_name, client_ids in clusters.items():
        cluster_buildings = cov_df[cov_df['building_id'].isin(client_ids)]
        # Primary use distribution
        use_dist = cluster_buildings['primary_use'].value_counts()
        # Temperature statistics
        temp_mean = cluster_buildings['air_temperature'].mean()
        temp_std = cluster_buildings['air_temperature'].std()
        composition_data.append({
            'cluster': cluster_name,
            'n_buildings': len(client_ids),
            'primary_uses': use_dist.to_dict(),
            'temp_mean': temp_mean,
            'temp_std': temp_std
        })
        print(f"\n{cluster_name}:")
        print(f" Buildings: {len(client_ids)}")
        print(f" Top building types: {use_dist.head(3).to_dict()}")
        print(f" Avg temperature: {temp_mean:.2f} ± {temp_std:.2f}")
    # Visualize cluster sizes
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Cluster size distribution
    cluster_names = [d['cluster'] for d in composition_data]
    cluster_sizes = [d['n_buildings'] for d in composition_data]
    axes[0].bar(range(len(cluster_names)), cluster_sizes, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Number of Buildings', fontsize=12)
    axes[0].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(cluster_names)))
    axes[0].set_xticklabels([f"C{i}" for i in range(len(cluster_names))], rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    # Temperature distribution across clusters
    temps = [d['temp_mean'] for d in composition_data]
    temp_stds = [d['temp_std'] for d in composition_data]
    axes[1].errorbar(range(len(cluster_names)), temps, yerr=temp_stds,
                     fmt='o', markersize=8, capsize=5, color='coral', ecolor='gray')
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Average Temperature (°C)', fontsize=12)
    axes[1].set_title('Temperature Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(cluster_names)))
    axes[1].set_xticklabels([f"C{i}" for i in range(len(cluster_names))], rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_analysis/01_cluster_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    return composition_data

def plot_cluster_timeseries(clusters, series_dict, synthetic_series, n_samples=10):
    """Plot sample time series from each cluster with synthetic data"""
    print("\n" + "="*80)
    print("TIME SERIES VISUALIZATION")
    print("="*80)
    n_clusters = len(clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 4*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    for idx, (cluster_name, client_ids) in enumerate(clusters.items()):
        ax = axes[idx]
        # Sample clients
        sample_ids = np.random.choice(client_ids,
                                      size=min(n_samples, len(client_ids)),
                                      replace=False)
        # Plot real client data
        for cid in sample_ids:
            series = series_dict[cid][:168]  # First week
            ax.plot(series, alpha=0.6, linewidth=1.5, label=f'Client {cid}')
        # Plot sample synthetic series
        for i in range(min(3, len(synthetic_series))):
            ax.plot(synthetic_series[i],
                    color='red', alpha=0.3, linewidth=2,
                    linestyle='--', label='Synthetic' if i == 0 else '')
        # Styling
        ax.set_title(f'{cluster_name} - Sample Time Series (n={len(client_ids)} buildings)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Energy Consumption', fontsize=10)
        ax.grid(True, alpha=0.3)
        # Only show legend for first few series
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:min(5, len(handles))],
                      labels[:min(5, len(labels))],
                      loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('cluster_analysis/02_cluster_timeseries_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved time series samples plot")

def plot_cluster_statistics(clusters, series_dict, resized_series):
    """Plot mean and variance statistics for each cluster"""
    print("\n" + "="*80)
    print("CLUSTER STATISTICS ANALYSIS")
    print("="*80)
    n_clusters = len(clusters)
    # Create subplots for each cluster
    for cluster_name, client_ids in clusters.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        # Gather all series for this cluster
        cluster_series = []
        for cid in client_ids:
            idx = list(series_dict.keys()).index(cid)
            cluster_series.append(resized_series[idx])
        cluster_series = np.array(cluster_series)
        # Compute statistics
        mean_series = np.mean(cluster_series, axis=0)
        std_series = np.std(cluster_series, axis=0)
        median_series = np.median(cluster_series, axis=0)
        q25 = np.percentile(cluster_series, 25, axis=0)
        q75 = np.percentile(cluster_series, 75, axis=0)
        # Plot 1: Mean with confidence bands
        axes[0, 0].plot(mean_series, color='darkblue', linewidth=2.5, label='Mean')
        axes[0, 0].fill_between(range(len(mean_series)),
                                mean_series - std_series,
                                mean_series + std_series,
                                alpha=0.3, color='lightblue', label='±1 Std Dev')
        axes[0, 0].fill_between(range(len(mean_series)),
                                mean_series - 2*std_series,
                                mean_series + 2*std_series,
                                alpha=0.15, color='lightblue', label='±2 Std Dev')
        axes[0, 0].set_title(f'{cluster_name} - Mean with Standard Deviation',
                             fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Energy Consumption')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        # Plot 2: Median with IQR
        axes[0, 1].plot(median_series, color='darkgreen', linewidth=2.5, label='Median')
        axes[0, 1].fill_between(range(len(median_series)), q25, q75,
                                alpha=0.4, color='lightgreen', label='IQR (25-75%)')
        axes[0, 1].set_title(f'{cluster_name} - Median with Interquartile Range',
                             fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Energy Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        # Plot 3: Variance over time
        axes[1, 0].plot(std_series**2, color='crimson', linewidth=2)
        axes[1, 0].set_title(f'{cluster_name} - Variance Over Time',
                             fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].grid(True, alpha=0.3)
        # Plot 4: Coefficient of Variation
        cv = (std_series / (np.abs(mean_series) + 1e-6)) * 100
        axes[1, 1].plot(cv, color='darkorange', linewidth=2)
        axes[1, 1].set_title(f'{cluster_name} - Coefficient of Variation (%)',
                             fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('CV (%)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        cluster_id = cluster_name.split('_')[-1]
        plt.savefig(f'cluster_analysis/03_cluster_{cluster_id}_statistics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved statistics plot for {cluster_name}")

def visualize_embeddings(encodings_scaled, labels, clusters):
    """Visualize cluster embeddings using t-SNE and PCA"""
    print("\n" + "="*80)
    print("EMBEDDING VISUALIZATION")
    print("="*80)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    # t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d_tsne = tsne.fit_transform(encodings_scaled)
    scatter1 = axes[0].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                               c=labels, cmap='tab20', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].set_title('t-SNE Projection of Client Embeddings', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Cluster ID', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    # PCA
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d_pca = pca.fit_transform(encodings_scaled)
    scatter2 = axes[1].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1],
                               c=labels, cmap='tab20', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'PCA Projection (Explained Var: {pca.explained_variance_ratio_.sum():.2%})',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Cluster ID', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_analysis/04_embedding_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved embedding visualization")
    return embeddings_2d_tsne, embeddings_2d_pca

def compute_cluster_separation(encodings_scaled, labels, clusters):
    """Compute inter-cluster and intra-cluster distances"""
    print("\n" + "="*80)
    print("CLUSTER SEPARATION ANALYSIS")
    print("="*80)
    K = len(clusters)
    # Compute centroids
    centroids = []
    for k in range(K):
        cluster_points = encodings_scaled[labels == k]
        centroids.append(cluster_points.mean(axis=0))
    centroids = np.array(centroids)
    # Inter-cluster distances
    inter_distances = cdist(centroids, centroids, metric='euclidean')
    # Intra-cluster distances (average distance to centroid)
    intra_distances = []
    for k in range(K):
        cluster_points = encodings_scaled[labels == k]
        centroid = centroids[k]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_distances.append(distances.mean())
    # Silhouette-like metric
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    silhouette = silhouette_score(encodings_scaled, labels)
    davies_bouldin = davies_bouldin_score(encodings_scaled, labels)
    calinski = calinski_harabasz_score(encodings_scaled, labels)
    print(f"\nClustering Quality Metrics:")
    print(f" Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
    print(f" Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f" Calinski-Harabasz Score: {calinski:.4f} (higher is better)")
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Inter-cluster distance heatmap
    im1 = axes[0].imshow(inter_distances, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Inter-Cluster Distance Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Cluster ID')
    for i in range(K):
        for j in range(K):
            if i != j:
                text = axes[0].text(j, i, f'{inter_distances[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Euclidean Distance')
    # Intra-cluster distances
    axes[1].bar(range(K), intra_distances, color='steelblue', alpha=0.7)
    axes[1].set_title('Average Intra-Cluster Distance', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Avg Distance to Centroid')
    axes[1].set_xticks(range(K))
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_analysis/05_cluster_separation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved cluster separation analysis")
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski,
        'inter_distances': inter_distances,
        'intra_distances': intra_distances
    }

def analyze_reconstruction_errors(ae_trend, ae_season, ae_resid,
                                  trend_real, seasonal_real, resid_real,
                                  labels, clusters, device):
    """Analyze autoencoder reconstruction quality per cluster"""
    print("\n" + "="*80)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("="*80)
    import torch
    # Compute reconstruction errors
    with torch.no_grad():
        trend_recon = ae_trend(torch.tensor(trend_real, dtype=torch.float32).to(device)).cpu().numpy()
        season_recon = ae_season(torch.tensor(seasonal_real, dtype=torch.float32).to(device)).cpu().numpy()
        resid_recon = ae_resid(torch.tensor(resid_real, dtype=torch.float32).to(device)).cpu().numpy()
    trend_errors = np.mean((trend_real - trend_recon)**2, axis=1)
    season_errors = np.mean((seasonal_real - season_recon)**2, axis=1)
    resid_errors = np.mean((resid_real - resid_recon)**2, axis=1)
    K = len(clusters)
    # Per-cluster errors
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = [
        ('Trend', trend_errors),
        ('Seasonal', season_errors),
        ('Residual', resid_errors)
    ]
    for idx, (comp_name, errors) in enumerate(components):
        ax = axes[idx]
        cluster_errors = [errors[labels == k] for k in range(K)]
        bp = ax.boxplot(cluster_errors, patch_artist=True, showmeans=True)
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_title(f'{comp_name} Component - Reconstruction Error by Cluster',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('MSE')
        ax.set_xticklabels(range(K))
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_analysis/06_reconstruction_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Print statistics
    for k in range(K):
        print(f"\nCluster {k}:")
        print(f" Trend MSE: {trend_errors[labels==k].mean():.6f} ± {trend_errors[labels==k