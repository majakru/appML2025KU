import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to all files
healthy_path = "type B pancreatic cell_normal_subset.h5ad"
disease_paths = [
    "type B pancreatic cell_type 2 diabetes mellitus_subset.h5ad",
    "type B pancreatic cell_type 1 diabetes mellitus_subset.h5ad",
    "type B pancreatic cell_endocrine pancreas disorder_subset.h5ad"
]

# Load healthy
adata_healthy = sc.read_h5ad(healthy_path)
print(f"Healthy shape: {adata_healthy.shape}")

# Load diseases
adata_diseases = [sc.read_h5ad(path) for path in disease_paths]
all_adatas = [adata_healthy] + adata_diseases

# Concatenate X matrices and collect labels
X_all = np.vstack([adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X for adata in all_adatas])
labels = ["normal"] * adata_healthy.shape[0]

for path, adata in zip(disease_paths, adata_diseases):
    disease_name = path.split("type A pancreatic cell_")[-1].split("_subset")[0]
    labels.extend([disease_name] * adata.shape[0])

#For Pancreatic D best or 10
# For B --> 10 
#A: 
# endocrime used 20 seened best


# PCA
N_PCA_COMPONENTS = 10
pca = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42)
X_pca_all = pca.fit_transform(X_all)

# Scaling
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_pca_all)

# Extract healthy portion for training
X_scaled_healthy = X_scaled_all[:adata_healthy.shape[0]]

# Autoencoder
input_dim = N_PCA_COMPONENTS
encoding_dim = 32
input_layer = Input(shape=(input_dim,))
"""
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)
"""

encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)



autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled_healthy, X_scaled_healthy, epochs=20, batch_size=16, shuffle=True, verbose=0)

encoder = Model(inputs=input_layer, outputs=encoded)

# Latent space for all samples
X_latent_all = encoder.predict(X_scaled_all)

# UMAP
umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_latent_all)

# Plotting
df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
df["disease"] = labels

#Save plot
cell_type = "type B pancreatic cell"
# Join disease names for uniqueness
disease_tags = "_".join(sorted(set(label.replace(" ", "_") for label in labels if label != "normal")))
filename = f"{cell_type.replace(' ', '_')}_{disease_tags}.png"
os.makedirs("saved_umaps", exist_ok=True)
filepath = os.path.join("saved_umaps", filename)

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="disease", alpha=0.6, s=10, palette="Set1")
plt.title(f"UMAP of Latent Space ({cell_type}, Multi-Disease)")

plt.legend(title="disease", oc='lower left', bbox_to_anchor=(0, 0), fontsize="small", title_fontsize="small", markerscale=2)

plt.savefig(filepath, dpi=300)
plt.show()

print(f"Saved UMAP to {filepath}")



# --- UMAP and Plot for Normal Cells Only ---

# --- Reuse existing UMAP output to plot only normal cells ---

# Extract rows from UMAP corresponding to normal cells
num_normal = adata_healthy.shape[0]
X_umap_normal_only = X_umap[:num_normal]
labels_normal = ["normal"] * num_normal

# Create DataFrame
df_normal = pd.DataFrame(X_umap_normal_only, columns=["UMAP1", "UMAP2"])
df_normal["disease"] = labels_normal

# Save normal-only plot
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_normal = f"{cell_type.replace(' ', '_')}_normal_only_{timestamp}.png"
filepath_normal = os.path.join("saved_umaps", filename_normal)

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_normal, x="UMAP1", y="UMAP2", hue="disease", alpha=0.7, s=12, palette="Set1")
plt.title(f"UMAP of Latent Space ({cell_type}, Normal Only)")
plt.tight_layout()
plt.savefig(filepath_normal, dpi=300)
plt.show()

print(f" Saved UMAP of normal cells to {filepath_normal}")
