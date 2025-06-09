import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.facecolor'] = '#e8e8e8'
plt.rcParams['axes.labelcolor'] = '565656'
plt.rcParams['xtick.color'] = '565656'
plt.rcParams['ytick.color'] = '565656'



# Load the data
electronics_df = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/electronic_descriptors.csv", index_col=0)
sterics_df = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/steric_descriptors.csv", index_col=0)

# Define ligand classes
ligand_classes = {
    "Symmetric Diamide": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L31"],
    "Asymmetric Diamide": ["L9", "L10"],
    "Phenanthroline": ["L11", "L12", "L13", "L14"],
    "Dipyrrazole": ["L15", "L16", "L17"],
    "6-Hydroxy Picolinohydrazide": ["L18", "L19", "L20", "L21"],
    "N,N-Diarylbenzene-1,2-diamine": ["L22", "L23", "L24"],
    "Amino-2-oxoacetic acid": ["L25", "L26", "L27"],
    "6‑Hydroxypicolinamide": ["L28", "L29", "L30", "L32"]
}

# Extract Boltzmann averaged rows
boltz_electronics = electronics_df.loc[electronics_df.index.str.contains('Boltzmann Average')]
boltz_sterics = sterics_df.loc[sterics_df.index.str.contains('Boltzmann Average')]

# Exclude ligands L21 and L24
boltz_electronics = boltz_electronics[~boltz_electronics.index.str.contains('L21|L24')]
boltz_sterics = boltz_sterics[~boltz_sterics.index.str.contains('L21|L24')]

# Combine and clean the data
boltz = pd.concat([boltz_electronics, boltz_sterics], axis=1)
boltz = boltz.dropna(axis=1, how='any')

# Convert to numpy array
X = boltz.to_numpy()

# Scale and filter low-variance features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = VarianceThreshold(threshold=1e-5).fit_transform(X_scaled)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', init='pca', random_state=42)
X_transformed = tsne.fit_transform(X_scaled)

# Map ligands to their classes
ligand_to_class = {ligand: cls for cls, ligands in ligand_classes.items() for ligand in ligands}
ligand_names = boltz.index.str.replace('_Boltzmann Average', '', regex=False)

# Assign classes
y = [ligand_to_class.get(name, 'Unknown') for name in ligand_names]

# Map classes to colors using Plasma colormap
unique_classes = sorted(set(y))
class_to_index = {cls: i for i, cls in enumerate(unique_classes)}
norm = Normalize(vmin=0, vmax=len(unique_classes) - 1)
plasma = cm.get_cmap('plasma')
y_colors = [plasma(norm(class_to_index[cls])) for cls in y]

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                     c=y_colors, s=100, alpha=0.8)

# Custom legend
for cls in unique_classes:
    ax.scatter([], [], color=plasma(norm(class_to_index[cls])), label=cls)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend()
ax.set_title('t-SNE of Ligand Descriptors', fontsize=12)
ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5, color='white', linewidth=1.0)
ax.set_axisbelow(True)
fig.tight_layout()
#plt.show()
plt.savefig("../data/ligand_clustering.png", dpi=400)

