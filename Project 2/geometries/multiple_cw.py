import copy

import matplotlib.pyplot as plt
import meshio
import numpy as np
import sys
sys.path.append("../scripts")
from utils import plot_mesh, extract_grid, transform_mesh, choose_rbf, save_mesh, \
    check_validity, stretch_cp, rotate_cp, inflate_cp

msh_template = meshio.read('LAA_template.msh')
template_points, template_cells, template_border = extract_grid(msh_template, flag_border=True)
radial_basis = choose_rbf(name='l2')
np.random.seed(2304)

name = 'CW_R'

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
# 2.359947648 / 1.681858113
#coordinates_final = inflate_cp(coordinates_final, coeff= np.sqrt(1.681858113 / 2.359947648))

#parameter
sigma = 0.5
new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                       final_cp=coordinates_final, radial_basis_function=radial_basis, **{'sigma_' : sigma})

inds = np.where(coordinates_final[:,0] != 0)[0]

n_sim = 10
mesh_list, coord_list = [], []
plt.figure()
for sim in range(n_sim):
    start = True

    while not check_validity(new_m) or start:

        coordinates_final_new = copy.deepcopy(coordinates_final)
        start = False
        noise_x = np.random.normal(loc=0, scale=0.02, size=inds.shape[0])
        noise_y = np.random.normal(loc=0, scale=0.02, size=inds.shape[0])
        random_scaling = np.random.normal(loc=1.1, scale=0.02, size=1)
        random_rotation = np.random.uniform(low = -np.pi/8, high=np.pi/8)

        coordinates_final_new[inds, 0] += noise_x
        coordinates_final_new[inds, 1] += noise_y

        coordinates_final_new = stretch_cp(coordinates_final_new, random_scaling)
        coordinates_final_new = rotate_cp(coordinates_final_new, random_rotation)

        new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                           final_cp=coordinates_final_new, radial_basis_function=radial_basis, **{'sigma_': sigma})

        print(f'Trying mesh n : {sim}')

    mesh_list.append(new_m)
    _, _, scatter_border = extract_grid(new_m, flag_border=True)
    plt.scatter(scatter_border[:, 0], scatter_border[:, 1])

    save_name = f'{name}/{name}_{sim}'
    save_mesh(new_m, save_name, n_control_points)

plt.show()

