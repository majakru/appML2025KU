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
normal_path = "Schwann cell_normal_subset.h5ad"
ST2D = "Schwann cell_endocrine pancreas disorder_subset.h5ad"

# Load both subsets
adata_normal = sc.read_h5ad(normal_path)
adata_t2d = sc.read_h5ad(ST2D)

# Optionally: limit cells to reduce memory if needed
# adata_normal = adata_normal[np.random.choice(adata_normal.obs_names, 2000, replace=False), :].copy()
# adata_t2d = adata_t2d[np.random.choice(adata_t2d.obs_names, 2000, replace=False), :].copy()

#1: PCA on normal cells only --> reduce number of genes 
#paramaters = genes 


"""
Sccells: comps is 200
"""
#This value varies depending the cel using --> i.e. much larger for type b
N_PCA_COMPONENTS = 200
pca = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42) 
X_normal_pca = pca.fit_transform(adata_normal.X) #reduction step --> size (# cells, PCA size(1000))

#Scaling data 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal_pca)

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
autoencoder.fit(X_normal_scaled, X_normal_scaled, epochs=20, batch_size=16, shuffle=True, verbose=0)

encoder = Model(inputs=input_layer, outputs=encoded)

#3:Encode all (normal + T2D)
X_normal_all = scaler.transform(pca.transform(adata_normal.X))
X_t2d_all = scaler.transform(pca.transform(adata_t2d.X))

X_latent_normal = encoder.predict(X_normal_all)
X_latent_t2d = encoder.predict(X_t2d_all)

#4: UMAP
X_combined = np.concatenate([X_latent_normal, X_latent_t2d], axis=0)

#Also change these labels (after the first plus sign ) --> FOR the legends labels
labels = ['normal'] * len(X_latent_normal) + ['Schwann cell endocrine pancreas disorder'] * len(X_latent_t2d)

umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_combined)

#5: Plot types
df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
df['disease'] = labels


# Automatically generate a unique filename from the input file

# ---- Update the cell type for plotting ---# i.e. the cell type = ...
cell_type = "Schwann cell"
# Extract disease type from the file path (e.g., "type 1 diabetes mellitus") --> Also change for plot
disease_type = ST2D.split("Schwann cell_")[-1].split("_subset")[0].replace(" ", "_")

# Create a safe filename
filename = f"{cell_type.replace(' ', '_')}_{disease_type}.png"
filepath = os.path.join("saved_umaps", filename)



# Saving UMAP outputs
os.makedirs("saved_umaps", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="disease", alpha=0.6, s=10, palette="Set1")
plt.title(f"UMAP of Latent Space ({cell_type}, {disease_type.replace('_', ' ')})")
plt.tight_layout()
plt.savefig(filepath, dpi=300)
plt.show()

print(f"✅ Saved UMAP to {filepath}")



