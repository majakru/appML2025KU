import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to your subset files
normal_path = "type B pancreatic cell_normal_subset.h5ad"
t2d_path = "type B pancreatic cell_type 2 diabetes mellitus_subset.h5ad"

# Load both subsets
adata_normal = sc.read_h5ad(normal_path)
adata_t2d = sc.read_h5ad(t2d_path)

# Optionally: limit cells to reduce memory if needed
# adata_normal = adata_normal[np.random.choice(adata_normal.obs_names, 2000, replace=False), :].copy()
# adata_t2d = adata_t2d[np.random.choice(adata_t2d.obs_names, 2000, replace=False), :].copy()

#1: PCA on normal cells only --> reduce number of genes 
#paramaters = genes 
N_PCA_COMPONENTS = 1000
pca = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42) #to save memory
X_normal_pca = pca.fit_transform(adata_normal.X) #reduction step --> size (# cells, PCA size(1000))

# Step 2: Train encoder on normal
input_dim = N_PCA_COMPONENTS
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal_pca, X_normal_pca, epochs=20, batch_size=16, shuffle=True, verbose=0)

encoder = Model(inputs=input_layer, outputs=encoded)

#3:Encode all (normal + T2D)
X_normal_all = pca.transform(adata_normal.X)
X_t2d_all = pca.transform(adata_t2d.X)

X_latent_normal = encoder.predict(X_normal_all)
X_latent_t2d = encoder.predict(X_t2d_all)

#4: UMAP
X_combined = np.concatenate([X_latent_normal, X_latent_t2d], axis=0)
labels = ['normal'] * len(X_latent_normal) + ['type 2 diabetes mellitus'] * len(X_latent_t2d)

umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_combined)

#5: Plot types
df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
df['disease'] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="disease", alpha=0.6, s=10, palette="Set1")
plt.title("UMAP of Latent Space (type B pancreatic cell)")
plt.tight_layout()
plt.show()
