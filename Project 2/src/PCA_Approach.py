## % import packages
from copy import deepcopy
import meshio
import numpy as np
import matplotlib.pyplot as plt
from PCA import pca_extended
from utils import extract_grid, choose_rbf, plot_mesh

#msh_template = meshio.read('geometries/LAA_template.msh')
#msh_chicken = meshio.read('geometries/LAA_CW.msh')

# %% import meshes, extract points
msh_CW = meshio.read('our_geometries/CW_19.msh')
msh_WS = meshio.read('our_geometries/WS_19.msh')

CW_points, CW_cells = extract_grid(msh_CW, flag_border=False)
WS_points, WS_cells = extract_grid(msh_WS, flag_border=False)

radial_basis = choose_rbf(name='l2')
dataset = np.empty(shape=(0, 5609*2))

plist = [CW_points, WS_points]
#p0 = np.logical_and(template_points[:,0] == 0, template_points[:,1] == 0)

# %% reshape dataset

for p in plist:
    datum = p.T.reshape(-1)
    dataset = np.concatenate([dataset, datum[np.newaxis,:]])

# %% pca

n_components = 2
pca = pca_extended(n_components=n_components)
pca.fit(dataset)
pca.plot_explained_variance()

# %% plot mean
mean = pca.mean_.reshape(2,5609).T
new_m = deepcopy(msh_CW)
new_m.points[:,0:2] = mean

plot_mesh(new_m)
plt.show()

# %%
comp = pca.components_[0].reshape(2,5609).T
new_m = deepcopy(msh_CW)
new_m.points[:,0:2] = comp

plot_mesh(new_m)
plt.show()
