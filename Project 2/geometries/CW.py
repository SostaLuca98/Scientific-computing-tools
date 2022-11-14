import matplotlib.pyplot as plt
import meshio
import numpy as np

import sys
sys.path.append("../scripts")
from utils import plot_mesh, extract_grid, transform_mesh, choose_rbf, save_mesh, check_validity, stretch_cp

msh_template = meshio.read('LAA_template.msh')
template_points, template_cells, template_border = extract_grid(msh_template, flag_border=True)


radial_basis = choose_rbf(name='l2')

# control points
#p0 = np.logical_and(template_points[:,0] == 0, template_points[:,1] == 0)
coordinates_init = np.array([
                             [1, 1.05],
                             [2.1, -0.15],
                             [2, -0.35],
                             [0.75, 0.15],
                             [2.2, -0.35], # extreme point
                             [0.37784, 0.12655],
                             [1.14838, 0.01569],
                             [1.69354, -0.23087],
                             [1.71713, 0.34794],
                             [1.31782, 0.79320],
                             [0.29379, 1.05358],
                             [0.13842, 1.01739],
                             [0.68486, 1.12679],
                             [1.80490, 0.23936],
                             [1.46963, 0.63484],
                             [1.97590, 0.01863]
                             ])

xy = np.array([np.zeros(shape=4), np.linspace(0,1,4)]).T
coordinates_init = np.concatenate([xy, coordinates_init])
n_control_points = coordinates_init.shape[0]

# control points displacements
coordinates_final = np.array([
                             [1.20, 1.33],
                             [2.5, - 0.50],
                             [2.15, - 0.60],
                             [1, 0.33],
                             [2.50, -0.67],
                             [0.56467, 0.28287],
                             [1.36491, 0.08960],
                             [1.82059, -0.38020],
                             [2.11249, 0.15354],
                             [1.59169, 0.89246],
                             [0.43032, 1.24411],
                             [0.21081, 1.11045],
                             [0.87335, 1.45462],
                             [2.22062, -0.01161],
                             [1.78994, 0.62295],
                             [2.40309, -0.30851]
                             ])

coordinates_final = np.concatenate([xy, coordinates_final])

#parameter
sigma = 0.5

new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                       final_cp=coordinates_final, radial_basis_function=radial_basis, **{'sigma_' : sigma})

CW_points, CW_cells, CW_border = extract_grid(new_m, flag_border=True)

x_min, x_max = 0, 2.8
y_min, y_max = -0.75, 1.5

# plt.subplot(1,2,1)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plot_mesh(msh_chicken, title='Chicken Wing', show=False)

# plt.subplot(1,2,2)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plot_mesh(new_m, show=False)
plt.scatter(coordinates_final[:,0], coordinates_final[:,1], color = 'r')
plt.show()

# # Saving procedure
name = 'CW'
#if check_validity(new_m):
#    save_mesh(new_m, name, n_control_points)
